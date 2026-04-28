# Git Workflow

This fork uses focused branches and PRs so the project history doubles as documentation.

## Branches

Create branches from `origin/main` unless a PR is intentionally stacked:

- `feat/<short-topic>` for new research or user-facing capability
- `fix/<short-topic>` for correctness fixes
- `perf/<short-topic>` for performance changes
- `docs/<short-topic>` for documentation-only work
- `ops/<short-topic>` for cluster, runner, CI, or release operations
- `test/<short-topic>` for test-only work
- `chore/<short-topic>` for repo hygiene and maintenance

Prefer normal branches in the active checkout. Use worktrees only when explicitly requested or when preserving multiple active dirty branches is necessary.

## Commits

Use the repo commit template:

```bash
git config commit.template .github/COMMIT_TEMPLATE.md
```

Commit subjects must use one of:

- `(feat) ...`
- `(fix) ...`
- `(perf) ...`
- `(docs) ...`
- `(ops) ...`
- `(test) ...`
- `(chore) ...`

Keep commits reviewable. If a commit mixes unrelated concerns, split it before opening a PR.

## Pull Requests

Open draft PRs while still gathering evidence. Mark them ready only after:

- The PR body has `What`, `Why`, `How to test`, and `Checklist` sections.
- Relevant docs or experiment logs are updated.
- Checks and cluster jobs are reported as passed, failed, blocked, or not run.

Use squash merge for feature-sized PRs unless preserving multiple meaningful commits helps future readers understand the work.

## Review Loop

Codex may monitor open PRs for comments, requested changes, and failing checks. It should classify feedback and prepare local fixes, but it must not push commits, resolve comments, or reply on GitHub without explicit approval.

When addressing review feedback:

1. Fetch current PR comments and checks.
2. Separate actionable requests from stale, informational, or conflicting comments.
3. Make a focused fix for each feedback cluster.
4. Run the smallest relevant verification.
5. Commit with the same prefix convention.
6. Push only after approval.
