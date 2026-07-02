---
name: develop-no-test
description: Like /develop, but skips test-writer invocation and post-impl verification (for repos without a test suite — see ADR-014 §12)
allowed-tools: Agent, Bash, Read, Edit, Write, Glob, Grep
argument-hint: "<feature description>"
---

Thin alias for `/develop --no-test`.

Invoke `/develop --no-test $ARGUMENTS` directly. The parent skill handles the no-test mode (see [ADR-014 §12](../../../docs/adr/014-spec-driven-tdd.md) and `.claude/skills/develop/SKILL.md` Phase 2.NT).
