---
name: agent-workflow
description: Plan-first workflow enforcement, task tracking, verification, and lessons capture. Use when Codex needs to plan non-trivial work, run verification before done, track tasks/todo.md, or record corrections in tasks/lessons.md.
---

# Agent Workflow

## Overview

Provide a plan-first, verification-driven workflow with task tracking and lessons capture to keep execution disciplined and repeatable.

## Workflow Defaults

- Enter plan mode for any non-trivial task (3+ steps or architectural decisions).
- Stop and re-plan immediately if something goes sideways.
- Use plan mode for verification steps, not just building.
- Write detailed specs up front to reduce ambiguity.

## Subagent Strategy

- If subagents are available, offload research, exploration, and parallel analysis to them.
- Use one tack per subagent for focused execution.
- Use subagents liberally to keep the main context clean for integration decisions.
- For complex problems, allocate more subagents to increase parallelism.

## Verification Discipline

- Never mark a task complete without proving it works.
- Diff behavior between the baseline and changes when relevant.
- Ask: "Would a staff engineer approve this?"
- Run tests, check logs, and demonstrate correctness.

## Elegance Check

- For non-trivial changes, pause and ask: "Is there a more elegant way?"
- If a fix feels hacky, implement the elegant solution using current knowledge.
- Skip this check for simple, obvious fixes to avoid over-engineering.
- Challenge your own work before presenting it.

## Autonomous Bug Fixing

- When given a bug report, fix it without asking for hand-holding.
- Point at logs, errors, and failing tests, then resolve them.
- Avoid context switching for the user.
- Fix failing CI tests without being told how.

## Task Management

- Plan first: write the plan to tasks/todo.md with checkable items.
- Verify plan: check in before starting implementation.
- Track progress: mark items complete as you go.
- Explain changes: add a short, high-level summary at each step.
- Document results: add a Review section to tasks/todo.md.
- Capture lessons: update tasks/lessons.md after corrections.

If tasks/todo.md does not exist, create it using this template:
```markdown


# Tasks

## Plan
- [ ] Describe the goal and success criteria
- [ ] List the major steps
- [ ] Identify risks and tests

## Progress
- [ ] Step 1
- [ ] Step 2
- [ ] Step 3

## Review
- [ ] Summarize what changed
- [ ] Note verification performed
- [ ] Capture follow-up items
```

If tasks/lessons.md does not exist, create it using this template:

```markdown
# Lessons

## YYYY-MM-DD
Correction:
Rule:
Prevention:
```

## Self-Improvement Loop

- After any correction from the user, add a lesson entry.
- Write a rule that prevents the same mistake.
- Iterate on lessons until the mistake rate drops.
- Review lessons at the start of a relevant project.

## Core Principles

- Simplicity first: make every change as simple as possible and minimize code impact.
- No laziness: find root causes and avoid temporary fixes.
- Minimal impact: change only what is necessary to avoid introducing bugs.
