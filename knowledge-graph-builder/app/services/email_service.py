"""SMTP email sending.

A thin async wrapper over ``aiosmtplib`` used to send transactional email —
currently only organization member-invitation emails. Configuration lives in
``app.core.config.Settings`` under the ``SMTP_*`` keys (Gmail-friendly: host
``smtp.gmail.com``, port 587, STARTTLS, a Google app password).

When SMTP is not configured (``SMTP_HOST`` unset), :func:`is_configured`
returns ``False`` and callers should skip sending rather than failing — an
invitation is still usable via its link even if no email went out.
"""

from __future__ import annotations

from email.message import EmailMessage

import aiosmtplib

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def is_configured() -> bool:
    """True when enough SMTP settings are present to attempt a send."""
    return bool(settings.SMTP_HOST and settings.SMTP_USERNAME)


def _from_address() -> str:
    """The envelope/From address — explicit SMTP_FROM, else the SMTP user."""
    return settings.SMTP_FROM or settings.SMTP_USERNAME or "no-reply@oraclous.local"


async def send_email(
    *,
    to: str,
    subject: str,
    body_text: str,
    body_html: str | None = None,
) -> None:
    """Send one email. Raises ``RuntimeError`` if SMTP is not configured and
    propagates ``aiosmtplib`` errors on a failed send — callers decide whether
    a send failure is fatal.

    Args:
        to: Recipient address.
        subject: Subject line.
        body_text: Plain-text body (always sent).
        body_html: Optional HTML alternative part.
    """
    if not is_configured():
        raise RuntimeError(
            "SMTP is not configured — set SMTP_HOST and SMTP_USERNAME to send email"
        )

    message = EmailMessage()
    message["From"] = _from_address()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body_text)
    if body_html:
        message.add_alternative(body_html, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USERNAME,
        password=settings.SMTP_PASSWORD,
        start_tls=settings.SMTP_USE_TLS,
    )
    logger.info("email sent to %s — %r", to, subject)
