---
title: "QA Validation Report — ORA-135 CORS Security Fix + Proxy IP Rate Limiting"
author: QA Evaluation Engineer
date: 2026-04-08
status: blocked
source: ORA-143
target: ORA-135 (branch fix/phase1/lead-backend/ora-135-cors-proxy-security)
---

# QA Validation Report — ORA-143

**Task:** QA VALIDATION — ORA-135 — CORS security fix + proxy IP rate limiting
**Branch:** `fix/phase1/lead-backend/ora-135-cors-proxy-security`
**Commit:** `2a1194b`
**Status:** BLOCKED — critical implementation gap found
**Date:** 2026-04-08

---

## Summary

Validation of ORA-135 cannot complete. The implementation references `rate_limit_exceeded_handler` from `app.core.rate_limiter` in two places, but the function was never defined in that module. This is a **P1 bug** — the auth-service cannot start in production on this branch.

Bug filed as **ORA-144** and assigned to Backend Lead Developer for triage and fix.

---

## Test Results

### 1. Primary Validation — `test_cors_proxy_security.py` (7 tests)

**Result: ❌ BLOCKED — 0 tests ran**

```
ERROR collecting tests/unit/test_cors_proxy_security.py
ImportError: cannot import name 'rate_limit_exceeded_handler' from 'app.core.rate_limiter'
  (/Users/reza/workspace/Oraclous/oraclous-data-studio/auth-service/app/core/rate_limiter.py)
```

**Root Cause:** Commit `2a1194b` updated `app/main.py` to import `rate_limit_exceeded_handler` from `app.core.rate_limiter` (replacing the private slowapi symbol `_rate_limit_exceeded_handler`). The test file also imports the same symbol. However, the function body was never added to `rate_limiter.py`. The module only defines `limiter` and `enforce_key_prefix_rate_limit`.

**Affected files:**
- `auth-service/app/core/rate_limiter.py` — missing `rate_limit_exceeded_handler`
- `auth-service/app/main.py:9` — broken import (app crash on startup)
- `auth-service/tests/unit/test_cors_proxy_security.py:19` — broken import

### 2. Regression Tests

| Test Suite | Tests | Result | Notes |
|---|---|---|---|
| `test_rate_limiting.py` | 9 | ✅ ALL PASS | No regression from ORA-135 changes |
| `test_jwt_sub_migration.py` | 13 | ✅ ALL PASS | No regression |
| `test_service_account_keys.py` | 5/6 pass | ⚠️ 1 PRE-EXISTING FAIL | `test_bcrypt_hash_verify` fails (see below) |

**Pre-existing failure — `test_bcrypt_hash_verify`:**
```
ValueError: bcrypt: no backends available to load
```
This is a passlib/bcrypt incompatibility with Python 3.12 (`crypt` module deprecated). **Not introduced by ORA-135** — confirmed via git history (file unchanged by commit `2a1194b`). Tracked separately.

---

## Code Review (Branch Analysis)

Items confirmed by direct code inspection since the app cannot be imported:

| Item | Status | Notes |
|---|---|---|
| `settings.ALLOWED_ORIGINS` present in config | ✅ | `list[str]` defaulting to `["http://localhost:8080"]` |
| `settings.TRUSTED_PROXY_IPS` present in config | ✅ | `list[str]` defaulting to `["127.0.0.1"]` |
| `main.py` uses `ALLOWED_ORIGINS` (not wildcard) | ✅ | `allow_origins=settings.ALLOWED_ORIGINS` |
| `main.py` restricts methods to GET/POST | ✅ | `allow_methods=["GET", "POST"]` |
| `ProxyHeadersMiddleware` added as outermost | ✅ | Added after CORSMiddleware (last = outermost in Starlette) |
| `rate_limit_exceeded_handler` defined | ❌ | **Missing** — never added to `rate_limiter.py` |

---

## Validation Checklist

- [ ] All 7 new tests in `test_cors_proxy_security.py` pass → **BLOCKED**
- [ ] `test_cors_wildcard_is_not_configured` passes → **BLOCKED**
- [ ] `test_proxy_rate_limit_buckets_by_real_ip` passes → **BLOCKED**
- [x] `test_rate_limiting.py` and `test_jwt_sub_migration.py` pass → ✅ PASS
- [ ] No regression in `test_service_account_keys.py` → ⚠️ 1 PRE-EXISTING FAIL (unrelated)
- [x] `ALLOWED_ORIGINS` present in Settings → ✅ (code review)
- [x] `TRUSTED_PROXY_IPS` present in Settings → ✅ (code review)

---

## Bug Filed

**ORA-144** — `BUG — rate_limit_exceeded_handler missing from rate_limiter.py (ORA-135 incomplete)`
- Priority: High
- Assigned to: Backend Lead Developer (for triage)
- Fix required: Define `rate_limit_exceeded_handler` in `app/core/rate_limiter.py` as a callable `(request, exc) -> Response` returning HTTP 429 without exposing rate-limit internals in the body

---

## Re-validation Requirements

Once ORA-144 is fixed, re-run:

```bash
cd oraclous-data-studio/auth-service
python3 -m pytest tests/unit/test_cors_proxy_security.py -v -m unit
python3 -m pytest tests/unit/test_rate_limiting.py tests/unit/test_jwt_sub_migration.py tests/unit/test_service_account_keys.py -v
```

All 7 tests in `test_cors_proxy_security.py` must pass and no regressions in the other suites.
