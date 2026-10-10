"""Identity adapters (Integration plan v0.3, section 9).

* ``DevCookieIdentity``: fixture/development/real_model only. Loopback synthetic identities carried in an
  HMAC-signed HttpOnly SameSite=Strict cookie. Cannot satisfy institutional identity readiness.
* ``TrustedProxyJwtIdentity``: student_release. Verifies a campus SSO proxy's signed JWT with a pinned
  algorithm, locally loaded key, issuer, audience, expiry, not-before and server-approved roles/course
  mapping (PyJWT). Unsigned identity headers are ignored. The SSO subject goes only to the
  ``PseudonymResolver``; model services never receive it.

Identity is never taken from the request body.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional, Protocol

COOKIE = "tutor_dev_identity"


@dataclass(frozen=True)
class Identity:
    owner_code: str          # pseudonymous; never sent to model services
    tenant_id: str
    course_ids: frozenset[str]
    synthetic: bool


class IdentityError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code  # UNAUTHORIZED | FORBIDDEN


class IdentityProvider(Protocol):
    institutional: bool

    def identify(self, headers: dict[str, str], cookies: dict[str, str]) -> Identity: ...


class PseudonymResolver(Protocol):
    def resolve(self, subject: str, tenant_id: str) -> str: ...


class HmacPseudonymResolver:
    """Development resolver: keyed hash of the subject. Production uses the separate campus service."""

    def __init__(self, key: bytes):
        self._key = key

    def resolve(self, subject: str, tenant_id: str) -> str:
        mac = hmac.new(self._key, f"{tenant_id}:{subject}".encode(), hashlib.sha256).hexdigest()
        return "p-" + mac[:32]


class DevCookieIdentity:
    institutional = False

    def __init__(self, key: bytes, tenant_id: str, course_id: str):
        self._key = key
        self.tenant_id = tenant_id
        self.course_id = course_id

    def _sign(self, user: str) -> str:
        return hmac.new(self._key, f"dev-identity:{user}".encode(), hashlib.sha256).hexdigest()

    def issue(self, user: str) -> str:
        raw = base64.urlsafe_b64encode(user.encode()).decode().rstrip("=")
        return f"{raw}.{self._sign(user)}"

    def identify(self, headers: dict[str, str], cookies: dict[str, str]) -> Identity:
        token = cookies.get(COOKIE)
        if not token or "." not in token:
            raise IdentityError("UNAUTHORIZED")
        raw, sig = token.rsplit(".", 1)
        try:
            user = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
        except (ValueError, UnicodeError):
            raise IdentityError("UNAUTHORIZED") from None
        if not hmac.compare_digest(sig, self._sign(user)):
            raise IdentityError("UNAUTHORIZED")
        owner = "dev-" + hmac.new(self._key, f"owner:{user}".encode(), hashlib.sha256).hexdigest()[:24]
        return Identity(owner_code=owner, tenant_id=self.tenant_id, course_ids=frozenset({self.course_id}),
                        synthetic=True)


class TrustedProxyJwtIdentity:
    institutional = True
    HEADER = "authorization"

    def __init__(self, *, public_key: str, algorithm: str, issuer: str, audience: str, tenant_id: str,
                 course_roles: dict[str, list[str]], resolver: PseudonymResolver, leeway: int = 30):
        if algorithm not in ("RS256", "ES256", "EdDSA"):
            raise ValueError("unsupported pinned algorithm")
        self._key = public_key
        self._alg = algorithm
        self._iss = issuer
        self._aud = audience
        self._tenant = tenant_id
        self._course_roles = {c: frozenset(r) for c, r in course_roles.items()}
        self._resolver = resolver
        self._leeway = leeway

    def identify(self, headers: dict[str, str], cookies: dict[str, str]) -> Identity:
        import jwt  # noqa: PLC0415 - pinned PyJWT[crypto]

        value = headers.get(self.HEADER, "")
        if not value.startswith("Bearer "):
            raise IdentityError("UNAUTHORIZED")
        try:
            claims = jwt.decode(value[7:], self._key, algorithms=[self._alg], audience=self._aud, issuer=self._iss,
                                leeway=self._leeway,
                                options={"require": ["exp", "nbf", "iat", "iss", "aud", "sub"]})
        except jwt.PyJWTError:
            raise IdentityError("UNAUTHORIZED") from None
        roles = claims.get("roles")
        if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
            raise IdentityError("FORBIDDEN")
        if claims.get("tenant") != self._tenant:
            raise IdentityError("FORBIDDEN")
        courses = frozenset(c for c, allowed in self._course_roles.items() if allowed & set(roles))
        if not courses:
            raise IdentityError("FORBIDDEN")
        owner = self._resolver.resolve(str(claims["sub"]), self._tenant)
        return Identity(owner_code=owner, tenant_id=self._tenant, course_ids=courses, synthetic=False)


def owner_ref(owner_code: str) -> str:
    """Minimal pseudonymous reference for audit rows."""
    return hashlib.sha256(f"owner-ref:{owner_code}".encode()).hexdigest()[:16]


def support_ref(owner_code: str, request_id: str) -> str:
    return hashlib.sha256(f"support:{owner_code}:{request_id}".encode()).hexdigest()[:20]


def build_identity(cfg, key: bytes) -> IdentityProvider:
    if cfg.identity.adapter == "dev_cookie":
        if cfg.profile == "student_release":
            raise ValueError("dev identities are not allowed in student_release")
        return DevCookieIdentity(key, cfg.tenant_id, cfg.course_id)
    j = cfg.identity.jwt
    return TrustedProxyJwtIdentity(public_key=cfg.path(j.public_key_file).read_text(encoding="ascii"),
                                   algorithm=j.algorithm, issuer=j.issuer, audience=j.audience,
                                   tenant_id=cfg.tenant_id, course_roles=j.course_roles,
                                   resolver=HmacPseudonymResolver(key), leeway=j.leeway_seconds)


def get_header(headers, name: str) -> Optional[str]:
    return headers.get(name)
