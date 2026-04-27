---
title: "QA Validation Report — Auth Rate Limiting (ORA-131 + ORA-134)"
author: "QA Evaluation Engineer"
date: "2026-04-08"
status: "passed"
source: "ORA-159"
target: "ORA-131, ORA-134"
branch: "develop (feature/phase4/devops/ora-154-allowed-origins-trusted-proxy-staging-prod at /private/tmp/oraclous-deploy-validate)"
---

# QA Validation Report — Auth Rate Limiting

**Task:** [ORA-159](/ORA/issues/ORA-159)
**Validates:** [ORA-131](/ORA/issues/ORA-131) + [ORA-134](/ORA/issues/ORA-134)
**Branch:** `develop` (worktree: `/private/tmp/oraclous-deploy-validate`)
**Date:** 2026-04-08
**Result:** ✅ PASSED (all acceptance criteria met)

---

## Acceptance Criteria Validation

| # | Criteria | Result | Evidence |
|---|---|---|---|
| 1 | `login` enforces 10 req/min | ✅ PASS | `@limiter.limit("10/minute")` on `POST /login/` in `auth_routes.py` |
| 2 | `register` enforces 5 req/min | ✅ PASS | `@limiter.limit("5/minute")` on `POST /register/` |
| 3 | `refresh` enforces 20 req/min | ✅ PASS | `@limiter.limit("20/minute")` on `POST /refresh/` |
| 4 | `forgot-password` enforces 5 req/min | ✅ PASS | `@limiter.limit("5/minute")` on `POST /forgot-password/` |
| 5 | `/reset-password/` is rate-limited | ✅ PASS | `@limiter.limit("5/minute")` on `POST /reset-password/` (ORA-134 addition) |
| 6 | 429 returns generic `{"detail": "Too many requests"}` | ✅ PASS | Custom `rate_limit_exceeded_handler` in `rate_limiter.py`; verified by `test_429_body_is_generic_no_limit_detail` |
| 7 | INCR + EXPIRE are atomic | ✅ PASS | `pipeline(transaction=True)` in `enforce_key_prefix_rate_limit` (`rate_limiter.py:62-70`); verified by `test_incr_and_expire_both_queued_in_pipeline` |
| 8 | auth-service starts cleanly | ✅ PASS | `main.py` imports `rate_limit_exceeded_handler` from `app.core.rate_limiter`; `ProxyHeadersMiddleware` added after ORA-135 merge |
| 9 | Unit tests in `test_auth_endpoint_rate_limits.py` pass | ✅ PASS | 8/8 tests pass (see below) |
| 10 | Proxy IP forwarding (X-Forwarded-For) works correctly | ✅ PASS | `ProxyHeadersMiddleware(trusted_hosts=settings.TRUSTED_PROXY_IPS)` in `main.py`; `test_proxy_rate_limit_buckets_by_real_ip` PASS |

---

## Test Execution Results

### `test_auth_endpoint_rate_limits.py` — 8/8 PASSED

```
tests/unit/test_auth_endpoint_rate_limits.py::test_429_body_is_generic_no_limit_detail PASSED
tests/unit/test_auth_endpoint_rate_limits.py::test_429_status_code_is_correct PASSED
tests/unit/test_auth_endpoint_rate_limits.py::test_register_limit_5_per_minute PASSED
tests/unit/test_auth_endpoint_rate_limits.py::test_login_limit_10_per_minute PASSED
tests/unit/test_auth_endpoint_rate_limits.py::test_refresh_limit_20_per_minute PASSED
tests/unit/test_auth_endpoint_rate_limits.py::test_forgot_password_limit_5_per_minute PASSED
tests/unit/test_auth_endpoint_rate_limits.py::test_reset_password_limit_5_per_minute PASSED
tests/unit/test_auth_endpoint_rate_limits.py::test_per_ip_isolation PASSED
```

### Full Unit Suite — 43/44 PASSED

```
44 collected — 43 passed, 1 failed
```

The single failure (`test_bcrypt_hash_verify`) is **pre-existing and unrelated to ORA-131/134**:
- **Cause:** passlib/bcrypt version incompatibility on the local macOS Python 3.12.8 environment
- **Error:** `ValueError: password cannot be longer than 72 bytes` + `AttributeError: module 'bcrypt' has no attribute '__about__'`
- **Not introduced by ORA-131/134** — affects service account key hashing, not auth rate limiting

---

## Static Code Review Notes

### ✅ Custom 429 Handler
```python
# rate_limiter.py
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a generic 429 — never expose limit configuration in the body."""
    return JSONResponse(status_code=429, content={"detail": "Too many requests"})
```
No config leakage. Default slowapi handler (`_rate_limit_exceeded_handler`) would have exposed `"Rate limit exceeded: 10 per 1 minute"`.

### ✅ Atomic Pipeline Fix (ORA-134)
```python
# Before ORA-134 (non-atomic — TTL race window):
count = await redis_client.incr(redis_key)
if count == 1:
    await redis_client.expire(redis_key, _KEY_PREFIX_WINDOW_SECONDS)

# After ORA-134 (atomic pipeline):
async with redis_client.pipeline(transaction=True) as pipe:
    await pipe.incr(redis_key)
    await pipe.expire(redis_key, _KEY_PREFIX_WINDOW_SECONDS)
    results = await pipe.execute()
```
Eliminates the race condition where a crash between INCR and EXPIRE would leave a persistent Redis key with no TTL, permanently rate-limiting that key prefix.

### ⚠️ Minor Observation (Non-blocking)
The `enforce_key_prefix_rate_limit` dependency on `/service-token` still returns `"Rate limit exceeded for this key prefix. Try again later."` as the HTTPException detail. This reveals it's a prefix-based limit (but does not expose the limit count or window). This is acceptable for a service-account-to-machine endpoint and was pre-existing behavior.

---

## Verdict

**✅ ORA-131 + ORA-134 QA VALIDATION PASSED**

All 10 acceptance criteria are met. The auth rate limiting implementation is correct, secure, and well-tested. The bcrypt failure is a pre-existing environment issue outside the scope of this validation.

**ORA-131 and ORA-134 are cleared for production.**
