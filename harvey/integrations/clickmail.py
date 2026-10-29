"""ClickMail provider — a REST API on top of Amazon SES.

ClickMail has no SMTP or IMAP: Harvey sends with ``POST /v1/emails`` and
reads replies from ClickMail's inbound store (``GET /v1/inbound``), which
receives mail for the verified domain through SES receipt rules.

.env:
    CLICKMAIL_API_KEY=     a project key with emails:send, inbound:read
    CLICKMAIL_BASE_URL=    the instance's API root, e.g. https://clickmail.example.com

The sending domain must be verified in the ClickMail project, the persona
email registered as an active sender, and inbound enabled with that address
(or a catch-all) registered — mail to an unregistered address is dropped.

SES reports bounces and complaints to ClickMail, not to the mailbox. So
every sent email is remembered in data/clickmail_sent.json and its status is
checked on each read until it is delivered; a bounced or complained one comes
back from get_replies as a bounce, which the Handler counts toward the
bounce-rate kill switch.
"""

import html
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr

import httpx

from harvey.integrations.mail_provider import (
    InboundMessage,
    MailProvider,
    SendResult,
    looks_like_bounce,
)
from harvey.paths import PROJECT_ROOT

logger = logging.getLogger("harvey.clickmail")

SENT_FILE = PROJECT_ROOT / "data" / "clickmail_sent.json"
TIMEOUT = httpx.Timeout(20.0, connect=10.0)
PAGE_SIZE = 50
MAX_INBOUND_PAGES = 10
# How far back the first read of a run looks, like the other providers' week.
FIRST_READ_DAYS = 7
# A sent email's status is followed this long, and no longer once delivered
# this long ago: SES reports nearly every bounce within minutes.
TRACK_DAYS = 7
DELIVERED_SETTLE_HOURS = 48
MAX_STATUS_CHECKS = 50
BOUNCE_STATUSES = ("bounce", "complain")


class ClickMailError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _pick(d: dict, *keys, default=""):
    """ClickMail's REST bodies are camelCase and its webhooks snake_case;
    take whichever the response carries."""
    for key in keys:
        value = d.get(key)
        if value not in (None, ""):
            return value
    return default


def _address(value) -> str:
    """An address from {"email": ...}, a list of those, or "Name <x@y>"."""
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = value.get("email") or value.get("address") or ""
    return parseaddr(str(value or ""))[1].lower()


def _items(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "emails", "results"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ClickMailError("inbound list: unexpected response shape")


def _text_to_html(body: str) -> str:
    return "<div>" + html.escape(body).replace("\n", "<br>\n") + "</div>"


def _load_sent() -> dict:
    try:
        data = json.loads(SENT_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"ClickMail: couldn't read {SENT_FILE.name} ({e}); starting it over.")
        return {}
    return data if isinstance(data, dict) else {}


def _save_sent(sent: dict) -> None:
    SENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SENT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(sent, indent=1), encoding="utf-8")
    tmp.replace(SENT_FILE)


class ClickMailProvider(MailProvider):
    name = "clickmail"

    def __init__(self, config, env, transport=None):
        self.config = config
        self.api_key = getattr(env, "clickmail_api_key", "")
        base = (getattr(env, "clickmail_base_url", "") or "").rstrip("/")
        self.base_url = base[:-3] if base.endswith("/v1") else base
        self._transport = transport  # httpx.MockTransport in tests
        # Where the last confirmed inbound read ended; None reads the week.
        self._read_after: datetime | None = None
        self._pending_after: datetime | None = None
        self._pending_done: set[str] = set()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=TIMEOUT, transport=self._transport)

    def _sender(self) -> tuple[str, str]:
        persona = getattr(self.config, "persona", None)
        return (getattr(persona, "name", "") or "", getattr(persona, "email", "") or "")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self._sender()[1])

    async def _request(self, method: str, path: str, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}", **kwargs.pop("headers", {})}
        async with self._client() as client:
            try:
                r = await client.request(method, f"{self.base_url}/v1{path}",
                                         headers=headers, **kwargs)
            except httpx.HTTPError as e:
                raise ClickMailError(f"{method} {path}: {e}") from e
        if r.status_code >= 400:
            detail = r.text[:200].replace("\n", " ")
            raise ClickMailError(f"{method} {path} answered {r.status_code}: {detail}")
        try:
            return r.json()
        except ValueError as e:
            raise ClickMailError(f"{method} {path}: response is not JSON") from e

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        thread_ref: str = "",
        in_reply_to: str = "",
    ) -> SendResult:
        name, address = self._sender()
        payload = {
            "from": f"{name} <{address}>" if name else address,
            "to": [to_email],
            "subject": subject,
            "text": body,
            "html": _text_to_html(body),
            "metadata": {"source": "harvey"},
        }
        if in_reply_to:
            if in_reply_to.startswith("<"):
                payload["inReplyTo"] = in_reply_to
                payload["references"] = [in_reply_to]
            else:
                payload["inReplyToEmailId"] = in_reply_to
        try:
            data = await self._request(
                "POST", "/emails", json=payload,
                headers={"Idempotency-Key": str(uuid.uuid4())},
            )
        except ClickMailError as e:
            logger.error(f"ClickMail send failed to {to_email}: {e}")
            return SendResult(ok=False, error=str(e)[:300])
        email_id = data.get("id") if isinstance(data, dict) else None
        if not email_id:
            # Accepted without an id: it may go out, so it isn't "failed".
            logger.warning(f"ClickMail accepted the email to {to_email} without an id.")
            return SendResult(ok=True)
        sent = _load_sent()
        sent[email_id] = {"to": to_email, "at": _iso(_now())}
        _save_sent(sent)
        return SendResult(ok=True, message_id=email_id, thread_ref=email_id)

    async def get_replies(self, limit: int = 50) -> list[InboundMessage]:
        """Replies received since the last confirmed read, then bounces and
        complaints among recently sent emails. Raises ClickMailError when
        ClickMail can't be read, so the Handler holds sending."""
        out = await self._read_inbound(limit)
        out.extend(await self._read_bounces())
        return out

    async def _read_inbound(self, limit: int) -> list[InboundMessage]:
        since = self._read_after or (_now() - timedelta(days=FIRST_READ_DAYS))
        _, mailbox = self._sender()
        newest = since
        listed = []
        for page in range(1, MAX_INBOUND_PAGES + 1):
            data = await self._request("GET", "/inbound", params={
                "to": mailbox, "dateFrom": _iso(since), "page": page, "limit": PAGE_SIZE,
            })
            items = _items(data)
            listed.extend(items)
            if len(items) < PAGE_SIZE:
                break
        else:
            raise ClickMailError(f"inbound has more than {MAX_INBOUND_PAGES} pages since {_iso(since)}")

        out = []
        for item in listed:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            if str(item.get("status", "")).upper() == "REJECTED":
                continue  # failed SES virus scanning; never shown
            received = _parse_time(_pick(item, "receivedAt", "received_at", "createdAt"))
            if received and received > newest:
                newest = received
            full = await self._request("GET", f"/inbound/{item['id']}")
            msg = self._to_message(full if isinstance(full, dict) else item)
            if msg:
                out.append(msg)
        # All of them, not the last `limit`: the marker moves past every one.
        self._pending_after = newest
        return out

    @staticmethod
    def _to_message(d: dict) -> InboundMessage | None:
        from_email = _address(_pick(d, "from", "fromEmail", "from_email"))
        if not from_email:
            return None
        subject = str(_pick(d, "subject"))
        body = str(_pick(d, "text", "textBody"))
        if not body:
            import re
            body = re.sub(r"<[^>]+>", " ", str(_pick(d, "html", "htmlBody")))
        message_id = str(_pick(d, "messageIdHeader", "messageId", "message_id")).strip()
        # The ClickMail id of the email this answers matches Harvey's outbox
        # (that's the message_id send_email returned); else the raw header.
        in_reply_to = str(_pick(d, "inReplyToEmailId", "in_reply_to_email_id",
                                "inReplyTo", "in_reply_to")).strip()
        return InboundMessage(
            provider_id=f"clickmail:{d.get('id')}",
            from_email=from_email,
            subject=subject,
            body=body[:5000],
            message_id=message_id,
            in_reply_to=in_reply_to,
            thread_ref=in_reply_to,
            date=str(_pick(d, "receivedAt", "received_at", "createdAt")),
            is_bounce=looks_like_bounce(from_email, subject),
        )

    async def _read_bounces(self) -> list[InboundMessage]:
        sent = _load_sent()
        now = _now()
        out = []
        expired = [eid for eid, rec in sent.items()
                   if (_parse_time(rec.get("at")) or now) < now - timedelta(days=TRACK_DAYS)]
        for eid in expired:
            sent.pop(eid)
        checks = sorted(sent.items(), key=lambda kv: kv[1].get("at", ""), reverse=True)
        for email_id, rec in checks[:MAX_STATUS_CHECKS]:
            try:
                data = await self._request("GET", f"/emails/{email_id}")
            except ClickMailError as e:
                logger.warning(f"ClickMail: couldn't check email {email_id}: {e}")
                continue
            status = str(data.get("status", "") if isinstance(data, dict) else "").lower()
            if any(s in status for s in BOUNCE_STATUSES):
                out.append(InboundMessage(
                    provider_id=f"clickmail-{status}:{email_id}",
                    from_email="mailer-daemon@clickmail",
                    subject=f"Undeliverable ({status}): {rec.get('to', '')}",
                    body=str(data.get("error") or status),
                    in_reply_to=email_id,
                    is_bounce=True,
                ))
                self._pending_done.add(email_id)
            elif status == "delivered":
                sent_at = _parse_time(rec.get("at")) or now
                if sent_at < now - timedelta(hours=DELIVERED_SETTLE_HOURS):
                    self._pending_done.add(email_id)
        if expired:
            _save_sent(sent)
        return out

    def confirm_read(self) -> None:
        if self._pending_after is not None:
            self._read_after = self._pending_after
        if self._pending_done:
            sent = _load_sent()
            for email_id in self._pending_done:
                sent.pop(email_id, None)
            _save_sent(sent)
            self._pending_done = set()

    async def test_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "CLICKMAIL_API_KEY missing from .env"
        if not self.base_url:
            return False, "CLICKMAIL_BASE_URL missing from .env"
        name, address = self._sender()
        if not address:
            return False, "persona.email is empty in harvey.yaml"
        try:
            senders = _items(await self._request("GET", "/senders"))
            await self._request("GET", "/inbound", params={"limit": 1})
        except ClickMailError as e:
            return False, str(e)[:300]
        active = {s.get("email", "").lower(): s for s in senders if isinstance(s, dict)}
        sender = active.get(address.lower())
        if not sender:
            return False, f"{address} is not a sender in this ClickMail project (Senders → add it)"
        if str(sender.get("status", "active")).lower() not in ("active", "") or sender.get("active") is False:
            return False, f"{address} is disabled in ClickMail"
        return True, f"ClickMail connected; sending and reading as {address}"
