# Synapse and Cortex Execution Plan

## Decision

The product boundary is:

- **Synapse**: the live Sophie runtime. This repository currently carries the
  `sophie-voice` name, but voice is one transport rather than its identity.
- **Cortex**: the durable personal continuity system currently in
  `/Users/mukeshkumar/play/synapse-v3`.
- **Sophie Core**: authentication, user/profile management, entitlements, and
  conventional application services.
- **Sophie Mob**: device UI, microphone, playback, and device context.

Do not rename either repository until the first versioned integration contract
is working. Both repositories currently have substantial uncommitted work and
the old Cortex repository contains many internal `Synapse` names.

## Responsibility Boundary

### Synapse Owns

- Realtime sessions and current conversation history.
- Voice transport, VAD, STT, TTS, playback lifecycle, and recovery.
- Sophie prompt compilation and current-session context.
- Tool declaration, execution loops, and tool result continuation.
- Model/provider selection and fallback.
- Runtime stance, response timing, and delivery decisions.
- Device and surface context supplied by trusted clients.
- Capturing a canonical session transcript.
- Deciding when a Cortex packet is relevant to the current turn.
- Degrading gracefully when Cortex is unavailable.

### Cortex Owns

- Durable source/session archive.
- Extraction of facts, events, operational items, entities, threads, states,
  decisions, questions, preferences, and observations.
- Reconciliation, correction, provenance, confidence, freshness, and lifecycle.
- Session summaries and cross-session continuity.
- Hybrid retrieval and compact factual packets.
- Open loops, entity profiles, expectations, and attention candidates.
- Background preparation of morning, evening, project, people, and follow-up
  packets.
- Producing candidates for proactive attention, never delivering them itself.

### Cortex Does Not Own

- Sophie's final prose, personality, or voice.
- Realtime provider selection.
- Live response timing or interruption policy.
- Device notifications or permission to interrupt.
- The active LLM conversation window.
- A second conversational runtime.

### Synapse Does Not Own

- Canonical durable truth.
- Long-term extraction or reconciliation.
- Cross-session memory ranking.
- Belief confidence calculated from long-term evidence.
- Durable task, entity, event, or relationship state.

## Contract 1: Session Ingestion

This is the first integration and is write-only from Synapse to Cortex.

### Envelope

```json
{
  "_v": "sophie.session.v1",
  "idempotency_key": "session:<user_id>:<session_id>:final",
  "user_id": "string",
  "session_id": "string",
  "conversation_id": "string|null",
  "channel": "voice",
  "started_at": "ISO-8601",
  "ended_at": "ISO-8601",
  "timezone": "Europe/London",
  "close_reason": "user_end|disconnect|timeout|server_shutdown|recovered",
  "messages": [
    {
      "id": "string",
      "role": "user|assistant",
      "content": "final transcript text",
      "timestamp": "ISO-8601",
      "turn_id": "string|null"
    }
  ],
  "metadata": {
    "providers": {},
    "tools": [],
    "recoveries": [],
    "cancellations": [],
    "latency_summary": {}
  }
}
```

Raw audio is excluded by default. Provider and reliability diagnostics are
stored as metadata and must not be treated as user-authored memory evidence.

### Synapse Work

1. Capture final user transcripts, final assistant transcripts, tool calls,
   tool results, recoveries, and cancellation reasons into one session record.
2. Finalize the envelope once per session close.
3. Write the envelope to a local durable delivery spool before attempting HTTP.
4. POST asynchronously to Cortex with bounded timeout and an internal service
   credential.
5. Retry with exponential backoff using the same idempotency key.
6. Mark delivery acknowledged only after Cortex returns an outbox/job ID.
7. Never block response generation, playback, microphone recovery, or socket
   close on Cortex.
8. Add a synthetic test that creates a multi-turn session without microphone
   input and verifies the exact emitted envelope.

### Cortex Work

1. Define a versioned request model for `sophie.session.v1`.
2. Require service authentication and validate that the supplied user is in
   scope for that caller.
3. Make final session ingestion idempotent at the job/outbox level, not only at
   the existing `source_archive` level.
4. Preserve the complete source envelope and expose its processing status.
5. Replace daemon-thread scheduling with a durable worker/job claim mechanism.
6. Return the same accepted job for repeated idempotency keys.
7. Keep extraction asynchronous and independent of the caller connection.
8. Add contract tests for duplicate delivery, worker restart, partial failure,
   malformed messages, and cross-user isolation.

### Acceptance Criteria

- Ending a voice session returns immediately even when Cortex is offline.
- Restarting Synapse eventually redelivers an unacknowledged session.
- Delivering the same envelope 10 times creates one source archive and one
  extraction job.
- Restarting Cortex during processing does not lose the job.
- The stored transcript exactly matches final user and assistant turns.
- No raw audio or provisional STT text becomes durable memory.

## Contract 2: Session Startup Handoff

Add this only after ingestion output has been inspected and accepted.

### Cortex Work

1. Build a deterministic `handover.v1` read model from the latest accepted
   session summary and canonical open state.
2. Return factual context, not instructions or final prose.
3. Enforce a strict token/character budget and attach evidence IDs.
4. Include freshness, confidence, and unknown/ambiguous markers.
5. Do not invoke a live LLM in the request path.
6. Cache the packet and refresh it when relevant durable state changes.

Suggested response:

```json
{
  "_v": "cortex.handover.v1",
  "generated_at": "ISO-8601",
  "last_session": {
    "ended_at": "ISO-8601",
    "summary": "string",
    "still_open": []
  },
  "active_threads": [],
  "important_changes": [],
  "evidence_ids": [],
  "warnings": []
}
```

### Synapse Work

1. Fetch once when opening a session, with a short timeout.
2. Cache the packet for the session.
3. Feed it into the existing prompt compiler as a bounded factual module.
4. Apply continuity decay: strongest on the first turn, reduced on the second,
   absent unless relevant afterward.
5. Start normally with no packet when Cortex is unavailable.
6. Log packet version, age, selected sections, and token estimate.

### Acceptance Criteria

- Cortex failure adds no more than the configured startup timeout.
- The first response can naturally continue the previous session.
- Stale or contradicted context is not presented as current truth.
- The packet stays within its fixed budget.
- The full Cortex store is never dumped into the model prompt.

## Contract 3: On-Demand Recall Tool

Add after startup handoff is stable.

### Cortex Work

- Expose a stable `recall.v1` evidence-packet contract.
- Return ranked items, source/evidence IDs, confidence, freshness, ambiguity,
  and `weak_recall`.
- Separate person, relationship, episodic, project, and operational lanes.
- Never return final Sophie prose.

### Synapse Work

- Register `search_memory` as a runtime-owned tool.
- Let the conversational model request it when memory is actually needed.
- Execute it server-side and continue the existing HF tool loop.
- Compile a bounded result packet rather than injecting raw retrieval output.
- Clarify or abstain when Cortex reports ambiguity or weak recall.
- Trace tool decision, query, latency, selected evidence, and answer grounding.

### Acceptance Criteria

- Ordinary conversation performs no Cortex recall.
- A memory question emits one visible runtime tool trace and a grounded answer.
- Weak recall produces uncertainty rather than invention.
- Tool timeout yields a natural response without breaking the voice session.

## Contract 4: Beliefs and Hypotheses

This is a Cortex evolution, not a prerequisite for integration.

Do not replace facts with beliefs. Keep distinct canonical primitive types:

- confirmed fact;
- observation;
- user-confirmed preference;
- extracted operational record;
- hypothesis/interpretation;
- current state;
- relationship state;
- unresolved question;
- correction/contradiction.

A hypothesis must carry:

- hypothesis text and scope;
- supporting and contradicting evidence IDs;
- confidence and confidence method;
- source agent/model/version;
- first observed, last supported, and last challenged timestamps;
- freshness/expiry policy;
- confirmation status;
- sensitivity and allowed-use policy.

Run hypothesis generation in shadow mode first. It may prepare context for
inspection but must not influence Sophie until precision, contradiction
handling, and deletion/correction semantics are evaluated.

## Contract 5: Prepared Cognition and Proactivity

This comes after reliable ingestion, handoff, and recall.

### Cortex

- Precompute factual packets for time-of-day, people, projects, open promises,
  relationship changes, habits, wins, concerns, and upcoming preparation.
- Produce `opportunity_candidate.v1` records containing evidence, confidence,
  priority, freshness, suggested timing windows, and sensitivity.
- Never send a notification or start a conversation.

### Synapse

- Decide whether and when a candidate may be delivered.
- Apply user preferences, interruption limits, recency, current activity,
  surface/device capability, quiet hours, repetition suppression, and safety.
- Choose notification, text, voice, natural in-session mention, defer, or drop.
- Record the outcome back to Cortex so candidates can be suppressed, revised,
  or reopened.

## Cross-Repository Engineering Rules

1. Contracts are versioned and additive. Never coordinate deployments through
   unversioned implicit Python object shapes.
2. Each contract has shared JSON fixtures checked by both repositories.
3. Cortex is never required for the realtime audio loop to remain healthy.
4. Synapse never writes directly to the Cortex database.
5. Cortex never returns final conversational prose.
6. User identity is established by Sophie Core/service authentication, not
   trusted from an unauthenticated mobile payload.
7. Every durable record retains provenance and supports correction/deletion.
8. Runtime traces and model diagnostics are not automatically memory evidence.
9. Expensive analysis runs asynchronously; request-time packets are cached or
   deterministic.
10. Every new intelligence feature first runs in trace/shadow mode.

## Execution Order

### Phase 0: Freeze Boundaries

**Synapse**

- Adopt the names in documentation without renaming the repository.
- Freeze the compact prompt compiler except for a future handover input slot.
- Preserve the current reliable half-duplex voice mode.

**Cortex**

- Mark `/v3/message`, `response_composer`, and the old `runtime/` package as
  legacy/experimental boundaries.
- Select `/v3/session/ingest` as the write foundation.

**Exit gate:** Both repositories agree on `sophie.session.v1` fixtures.

### Phase 1: Durable Session Write

Implement Contract 1 in both repositories.

**Exit gate:** offline/restart/idempotency tests pass and a real voice session
appears once in Cortex with correct extraction outputs.

### Phase 2: Observe Cortex Output

- Run at least 20 representative sessions through extraction.
- Review summaries, facts, entities, threads, open items, emotional-state
  claims, false positives, duplicates, and contradictions.
- Establish precision-focused evaluation fixtures.

**Exit gate:** accepted extraction quality and explicit list of fields safe to
surface.

### Phase 3: Startup Handoff

Implement Contract 2.

**Exit gate:** compact-versus-no-handoff listening tests show better continuity
without prompt bloat or startup fragility.

### Phase 4: Recall Tool

Implement Contract 3 through the existing HF runtime-owned tool loop.

**Exit gate:** grounded recall, weak-recall, timeout, and ambiguity tests pass.

### Phase 5: Deployment Foundation

**Synapse**

- CPU/cloud-provider container, health/readiness, durable delivery spool,
  graceful shutdown, TLS WebSocket ingress, provider secret management, and
  pipeline capacity metrics.

**Cortex**

- PostgreSQL/pgvector deployment, durable worker, migrations, backups,
  authentication, processing metrics, and dead-letter handling.

**Sophie Core**

- Issue short-lived runtime/session credentials and authoritative user/profile
  identity.

**Exit gate:** TestFlight client completes sessions against staging; Cortex
outage does not break conversation; Synapse restart does not lose transcript.

### Phase 6: Cognitive Evolution

- Shadow hypotheses.
- Prepared packets.
- Opportunity candidates.
- Stance hysteresis and bounded attention bursts in Synapse.
- True WebRTC/AEC barge-in only after the stable half-duplex product is
  operating in production.

## Immediate Next Work

The next implementation task is Phase 0 plus the contract fixture:

1. Define `sophie.session.v1` as JSON Schema and realistic fixtures.
2. Add contract validation tests in both repositories.
3. Trace Synapse final transcript events into an in-memory session envelope.
4. Do not call Cortex yet.
5. Inspect the resulting envelope from several synthetic and physical sessions.

Only then add the durable spool and Cortex HTTP ingestion. This keeps the first
change observable and prevents debugging transcript capture, network delivery,
idempotency, and extraction simultaneously.
