# Batch Patches Before Verify — Workflow Pattern

## Problem

A common agent anti-pattern in kernel optimization is the patch → remote_verify →
patch → remote_verify loop. Each `remote_verify` is an expensive SSH round-trip
(compile + test + bench on physical hardware). The agent wastes dozens of turns
verifying after every minor patch instead of batching all changes first.

## Root Cause

The agent's training bias toward "test after every change" conflicts with the
hardware verification cost. In simulation (cannsim), tests are free and instant.
On hardware, each verify costs 30-120 seconds of wall time.

## Solution

The profiling skill instructs the agent to:
1. Apply ALL patches for the current optimization
2. Call `remote_verify` **once**
3. If it fails, diagnose and batch fixes
5. Only re-verify after all fixes are applied

Calling `remote_verify` more than 2-3 times per kernel is almost always wasteful.

## Observed Impact

In the l1_25_Swish optimization session, the agent called `remote_verify` **7 times**
across **97 GPT turns**. The kernel was correct after the first fix (turn ~80).
The remaining 6 verifies and 17 subsequent turns were wasted on unnecessary
re-verification and over-engineering.

With batch-first workflow, this could have been done in ~30 turns with 2 verifies.
