---
name: simple-dev-no-test
description: Like /simple-dev, but skips test-writer invocation and post-impl verification (for repos without a test suite — see ADR-014 §12)
allowed-tools: Agent, Bash, Read, Write, Glob, TaskCreate, TaskUpdate
argument-hint: "<feature description>"
---

Thin alias for `/simple-dev --no-test`.

Invoke `/simple-dev --no-test $ARGUMENTS` directly. The parent skill handles the no-test mode (see [ADR-014 §12](../../../docs/adr/014-spec-driven-tdd.md) and `.claude/skills/simple-dev/SKILL.md`).
