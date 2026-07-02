---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "<plan, design, or topic to interview about>"
---

Topic to interview about: $ARGUMENTS

(If the above is empty, ask the user once what they want to be grilled on, then proceed.)

Work through the topic in three phases:

1. Discuss what we want to build (feature, system, etc.).
2. Discuss the high-level design choices.
3. Interview the user relentlessly about every aspect until you reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one by one. For each question, provide your recommended answer. Ask one to four questions at a time.

If a question can be answered by exploring the codebase, explore the codebase instead of asking.
