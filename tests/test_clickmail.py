"""ClickMail provider: sending through the REST API, threaded replies,
reading the inbound store from a marker, bounces from email status, and
failing loudly."""

import json

import httpx
import pytest

from harvey.integrations import clickmail
from harvey.integrations.clickmail import ClickMailError, ClickMailProvider

pytestmark = pytest.mark.asyncio(loop_scope="function")


class Env:
    clickmail_api_key = "ck_test_123"
    clickmail_base_url = "https://mail.example.test/v1/"


class Persona:
    name = "Ana Example"
    email = "ana@sender.example"


class Config:
    persona = Persona()


class Api:
    """A tiny fake of the ClickMail endpoints Harvey uses."""

    def __init__(self):
        self.requests = []
        self.inbound = []        # full inbound emails, newest first
        self.status = {}         # email id -> status
        self.fail = {}           # (method, path) -> status
        self.sent = 0

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        path, method = request.url.path, request.method
        if (method, path) in self.fail:
            return httpx.Response(self.fail[(method, path)], json={"error": "boom"})
        if path == "/v1/emails" and method == "POST":
            self.sent += 1
            return httpx.Response(200, json={"id": f"email_{self.sent}", "status": "queued"})
        if path.startswith("/v1/emails/"):
            eid = path.rsplit("/", 1)[1]
            return httpx.Response(200, json={"id": eid, "status": self.status.get(eid, "sent")})
        if path == "/v1/inbound":
            page = int(request.url.params.get("page", 1))
            limit = int(request.url.params.get("limit", 50))
            listed = [{k: v for k, v in m.items() if k not in ("text", "html")}
                      for m in self.inbound]
            return httpx.Response(200, json={"data": listed[(page - 1) * limit:page * limit]})
        if path.startswith("/v1/inbound/"):
            iid = path.rsplit("/", 1)[1]
            for m in self.inbound:
                if m["id"] == iid:
                    return httpx.Response(200, json=m)
        if path == "/v1/senders":
            return httpx.Response(200, json={"data": [
                {"email": "ana@sender.example", "status": "active"}]})
        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(clickmail, "SENT_FILE", tmp_path / "clickmail_sent.json")
    return Api()


def provider(api):
    return ClickMailProvider(Config(), Env(), transport=httpx.MockTransport(api))


def body(request):
    return json.loads(request.content)


async def test_send_posts_plain_text_as_the_persona(api):
    p = provider(api)
    result = await p.send_email("lead@prospect.example", "Hola", "Línea 1\n<b>2</b>")
    assert result.ok and result.message_id == "email_1"
    req = api.requests[-1]
    assert str(req.url) == "https://mail.example.test/v1/emails"
    assert req.headers["Authorization"] == "Bearer ck_test_123"
    assert req.headers["Idempotency-Key"]
    sent = body(req)
    assert sent["from"] == "Ana Example <ana@sender.example>"
    assert sent["to"] == ["lead@prospect.example"]
    assert sent["text"] == "Línea 1\n<b>2</b>"
    assert "&lt;b&gt;" in sent["html"]
    assert json.loads(clickmail.SENT_FILE.read_text())["email_1"]["to"] == "lead@prospect.example"


async def test_reply_threads_on_the_raw_message_id(api):
    p = provider(api)
    await p.send_email("lead@prospect.example", "Re: Hola", "Dale", in_reply_to="<abc@prospect.example>")
    sent = body(api.requests[-1])
    assert sent["inReplyTo"] == "<abc@prospect.example>"
    assert sent["references"] == ["<abc@prospect.example>"]


async def test_send_failure_is_reported_not_raised(api):
    api.fail[("POST", "/v1/emails")] = 403
    result = await provider(api).send_email("lead@prospect.example", "Hola", "x")
    assert not result.ok and "403" in result.error
    assert not clickmail.SENT_FILE.exists()


async def test_reads_replies_and_moves_the_marker_only_on_confirm(api):
    api.inbound = [{
        "id": "in_1", "fromEmail": "Lead@Prospect.example", "fromName": "Lead",
        "subject": "Re: Hola", "text": "Me interesa", "messageIdHeader": "<r1@prospect.example>",
        "inReplyToEmailId": "email_1", "receivedAt": "2030-01-01T10:00:00.000Z",
    }]
    p = provider(api)
    replies = await p.get_replies()
    assert len(replies) == 1
    msg = replies[0]
    assert msg.from_email == "lead@prospect.example"
    assert msg.body == "Me interesa"
    assert msg.in_reply_to == "email_1"
    assert msg.message_id == "<r1@prospect.example>"
    assert msg.provider_id == "clickmail:in_1"
    assert not msg.is_bounce
    listing = [r for r in api.requests if r.url.path == "/v1/inbound"][0]
    assert listing.url.params["to"] == "ana@sender.example"

    await p.get_replies()  # not confirmed: still reads the past week
    again = [r for r in api.requests if r.url.path == "/v1/inbound"][-1]
    assert not again.url.params["dateFrom"].startswith("2030")
    p.confirm_read()
    await p.get_replies()
    after = [r for r in api.requests if r.url.path == "/v1/inbound"][-1]
    assert after.url.params["dateFrom"].startswith("2030-01-01T10:00:00")


async def test_inbound_read_failure_raises(api):
    api.fail[("GET", "/v1/inbound")] = 500
    with pytest.raises(ClickMailError):
        await provider(api).get_replies()


async def test_rejected_inbound_is_skipped(api):
    api.inbound = [{"id": "in_v", "status": "REJECTED", "from": "x@prospect.example",
                    "subject": "virus", "text": "x"}]
    assert await provider(api).get_replies() == []


async def test_bounced_email_comes_back_as_a_bounce_once_confirmed(api):
    p = provider(api)
    await p.send_email("gone@prospect.example", "Hola", "x")
    api.status["email_1"] = "bounced"
    out = await p.get_replies()
    bounces = [m for m in out if m.is_bounce]
    assert len(bounces) == 1
    assert bounces[0].in_reply_to == "email_1"
    assert "gone@prospect.example" in bounces[0].subject
    p.confirm_read()
    assert "email_1" not in json.loads(clickmail.SENT_FILE.read_text())
    assert [m for m in await p.get_replies() if m.is_bounce] == []


async def test_complaint_counts_as_a_bounce(api):
    p = provider(api)
    await p.send_email("angry@prospect.example", "Hola", "x")
    api.status["email_1"] = "complained"
    assert [m.in_reply_to for m in await p.get_replies() if m.is_bounce] == ["email_1"]


async def test_test_connection_checks_the_sender(api):
    ok, detail = await provider(api).test_connection()
    assert ok, detail

    class Other(Config):
        class persona:
            name = "X"
            email = "nobody@sender.example"
    p = ClickMailProvider(Other(), Env(), transport=httpx.MockTransport(api))
    ok, detail = await p.test_connection()
    assert not ok and "not a sender" in detail


class LockedEnv(Env):
    clickmail_domain = "Sender.example"


class OtherDomainEnv(Env):
    clickmail_domain = "other.example"


async def test_domain_lock_allows_its_own_domain(api):
    p = ClickMailProvider(Config(), LockedEnv(), transport=httpx.MockTransport(api))
    assert (await p.send_email("lead@prospect.example", "Hola", "x")).ok
    assert (await p.test_connection())[0]


async def test_domain_lock_refuses_another_domain(api):
    p = ClickMailProvider(Config(), OtherDomainEnv(), transport=httpx.MockTransport(api))
    result = await p.send_email("lead@prospect.example", "Hola", "x")
    assert not result.ok and "other.example" in result.error
    assert api.sent == 0
    with pytest.raises(ClickMailError):
        await p.get_replies()
    assert not (await p.test_connection())[0]


async def test_mail_to_other_mailboxes_of_the_project_is_skipped(api):
    api.inbound = [
        {"id": "in_a", "fromEmail": "a@prospect.example", "toEmails": ["ana@sender.example"],
         "subject": "Re: Hola", "text": "Sí"},
        {"id": "in_b", "fromEmail": "b@prospect.example", "toEmails": ["support@sender.example"],
         "subject": "Ayuda", "text": "Otro buzón"},
    ]
    replies = await provider(api).get_replies()
    assert [m.from_email for m in replies] == ["a@prospect.example"]


async def test_factory_builds_clickmail():
    from harvey.integrations.mail_provider import NATIVE_PROVIDERS, get_mail_provider

    class C(Config):
        class channels:
            class email:
                provider = "clickmail"
    assert "clickmail" in NATIVE_PROVIDERS
    assert isinstance(get_mail_provider(C(), Env()), ClickMailProvider)
