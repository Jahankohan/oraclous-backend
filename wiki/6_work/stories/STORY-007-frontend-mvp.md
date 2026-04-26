---
id: STORY-007
title: "Frontend MVP: 4 working screens with full backend integration"
type: feature
layer: cross-cutting
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: []
tasks: []
decisions: []
---

# STORY-007: Frontend MVP

## Summary

The backend has 14 endpoint groups, 25+ services, auth, MCP, RAGAS evaluation, federation —
but no user can interact with it without API knowledge. The React + TypeScript frontend
(`oraclous-visual-flow-main/`) is scaffolded but not integrated with the backend. This
story delivers 4 working screens: Dashboard, Graph Detail, Chat, and Graph Explorer.
Without a frontend, the platform is a library that happens to have HTTP endpoints.

## Problem Statement

- `oraclous-visual-flow-main/` exists (React 18, TypeScript, Vite, React Flow, shadcn/ui)
- No API calls are wired; pages show mock or empty data
- Login, graph creation, document upload, chat — all exist as UI scaffolds with no backend connection
- Every competitor Oraclous is architecturally ahead of is easier to use because they have UIs or CLIs

## Goals

- [ ] Wire Login screen to `/auth/login` (JWT stored in cookie/localStorage)
- [ ] Wire Dashboard to `GET /api/v1/graphs` — show graph cards with node/relationship counts
- [ ] Wire "Create Graph" modal to `POST /api/v1/graphs`
- [ ] Wire Graph Detail: document list, file upload (multipart ingest), real-time job progress polling
- [ ] Wire Chat tab: message history, retriever type selector, source citations display, grounding indicator
- [ ] Wire Graph Explorer: node-link visualization, properties panel on click, filter by entity type

## Non-Goals

- Admin UI (tenant management, service account management)
- Visual pipeline builder (React Flow is scaffolded but out of scope for MVP)
- Mobile/responsive layout
- Community detection UI or federation management UI

## Acceptance Criteria

- [ ] User can sign in via Login screen and reach Dashboard without API client
- [ ] User can create a graph, upload a document, and see ingestion progress bar complete
- [ ] User can chat with a graph and see source citations alongside the answer
- [ ] User can click a node in Graph Explorer and see its properties
- [ ] No hardcoded URLs — all endpoints read from `VITE_API_BASE_URL` env var
- [ ] No CORS errors in browser console on default docker-compose setup

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | WebSocket or polling for ingestion progress? Backend supports both. | engineering | open |
| 2 | Graph Explorer: React Flow (existing) or d3-force? | engineering | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 11 (Frontend MVP, pp. 757-819)
- Frontend dir: `oraclous-visual-flow-main/`
- API integration table in spec covers all 9 required endpoints
- React Query already in frontend dependencies — use for API state management
- Estimated effort: 3-4 weeks
