# Backend Knowledge Base — oraclous-data-studio

Technical knowledge base for the Oraclous backend platform.

## Structure

```
knowledge-base/
├── specs/                 Technical specifications (before implementation)
│   ├── data-models/       Schema designs, ontologies (Data Engineer)
│   ├── api-contracts/     API designs, MCP tools (AI Integration)
│   └── integrations/      Framework integrations, connector specs
├── reviews/               Code review and security audit records
│   ├── security/          Security review findings
│   └── architecture/      Architecture review reports
├── qa/                    Quality assurance artifacts
│   ├── test-plans/        Test strategies and plans
│   ├── evaluation-reports/ KG quality, RAGAS scores
│   └── benchmarks/        Performance benchmarks
└── operations/            Infrastructure and operations
    ├── runbooks/          Operational procedures
    └── ci-cd/             Pipeline documentation
```

## Writing a Spec

1. Use `specs/_TEMPLATE.md` as your starting point
2. Create a branch: `spec/<phase>/<agent-slug>/<topic>`
3. Write the spec in the appropriate subdirectory
4. Create a PR — Backend Lead reviews for implementability
5. Implementation MUST NOT start until spec PR is merged

## Writing a Review

1. Use `reviews/_TEMPLATE.md`
2. Create a branch: `review/<phase>/<agent-slug>/<topic>`
3. Reference the implementation issue (ORA-XX) being reviewed
4. Create a PR — CTO reviews architecture reviews, Backend Lead reviews security reviews
