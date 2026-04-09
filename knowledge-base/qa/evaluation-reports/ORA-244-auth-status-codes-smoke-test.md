---
title: "QA Smoke Test Report — Auth HTTP Status Codes (ORA-244)"
author: QA Evaluation Engineer
date: 2026-04-09
status: PASSED
source: fix/phase1/sr-backend/ora-224-auth-status-codes
target: ORA-226 / PR #16
---

# QA Smoke Test Report — Auth HTTP Status Code Fixes

**Task:** [ORA-244](/ORA/issues/ORA-244)
**Parent Fix:** [ORA-226](/ORA/issues/ORA-226)
**Bug:** [ORA-224](/ORA/issues/ORA-224)
**Branch:** `fix/phase1/sr-backend/ora-224-auth-status-codes`
**PR:** [#16](https://github.com/Jahankohan/oraclous-backend/pull/16) → `develop`
**Date:** 2026-04-09
**Verdict:** ✅ **PASSED — Clear to merge**

---

## Test Scenarios

| Endpoint | Scenario | Expected | Result |
|---|---|---|---|
| `POST /register/` | New user success | **201 Created** | ✅ PASS |
| `POST /register/` | Duplicate email | **409 Conflict** + `{"detail": "Email already registered"}` | ✅ PASS |
| `POST /login/` | Wrong credentials | **401 Unauthorized** + `{"detail": "Incorrect email/username or password"}` | ✅ PASS |

---

## Unit Test Results

### New Tests (ORA-226)

```
tests/unit/test_auth_status_codes.py::test_register_route_declares_201                  PASSED
tests/unit/test_auth_status_codes.py::test_create_user_duplicate_raises_409              PASSED
tests/unit/test_auth_status_codes.py::test_authenticate_user_wrong_password_returns_none PASSED
tests/unit/test_auth_status_codes.py::test_login_route_raises_401_on_bad_credentials     PASSED

Result: 4/4 passed
```

### Regression Suite (Full Unit Suite)

```
tests/unit/test_auth_endpoint_rate_limits.py    8/8  passed
tests/unit/test_auth_status_codes.py            4/4  passed
tests/unit/test_cors_proxy_security.py          7/7  passed
tests/unit/test_jwt_sub_migration.py           14/14 passed
tests/unit/test_rate_limiting.py                8/8  passed
tests/unit/test_service_account_keys.py         7/7  passed

Total: 48/48 passed — 0 regressions
```

---

## Code Review Summary

### `auth_routes.py`
- `/register/` decorator updated: `status_code=201` ✅
- `/login/` raises `HTTPException(status_code=401)` on `None` return from service ✅ (was `400`)
- No unintended side effects on other routes

### `auth_service.py`
- `create_user()` raises `HTTPException(status_code=409, detail="Email already registered")` when user already exists ✅ (was not raising — returned 200)
- All other service methods unchanged

### `tests/unit/test_auth_status_codes.py`
- 4 well-scoped unit tests covering all 3 scenarios + service contract
- Uses mocks correctly; rate limiter bypassed via patch for route-layer test
- No DB or Redis dependency — fully isolated

---

## CI Reference

| Run | Status |
|---|---|
| PR run `24170169791` | ✅ passed |
| Push run `24170163233` | ✅ passed |
| Local unit run (2026-04-09) | ✅ 48/48 |

---

## Security Check

- No internal error details leaked in 409 / 401 response bodies
- Error messages use generic spec-compliant strings
- No stack traces or DB info in error responses

---

## Acceptance Criteria Checklist

- [x] `POST /register/` success returns 201
- [x] `POST /register/` duplicate returns 409 `{"detail": "Email already registered"}`
- [x] `POST /login/` wrong credentials returns 401 `{"detail": "Incorrect email/username or password"}`
- [x] Unit tests (4 new) confirmed passing locally
- [x] No regressions in existing auth flows (48/48 pass)
- [x] Response body detail strings match spec
- [x] Security: no internal error leakage

---

## Decision

**✅ QA APPROVED** — All acceptance criteria met. PR #16 is clear to merge to `develop`.

Backend Lead Developer may proceed with merge.
