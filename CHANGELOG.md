# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `NimbleWebSearchAgent`, a native `agent_framework.BaseAgent` backed by
  Nimble's Agent API V2 Web Search Agent.
- Generated-agent and persistent-agent modes.
- Multi-turn session resumability via `previous_interaction_id`.
- Framework-compatible `stream=True` support (single completed-answer update).
- Effort-tier gating: `effort="max"` (a coming-soon, custom-budget tier) is
  rejected by default with an actionable error, or auto-substituted with
  `effort="x-high"` when `gate_policy="degrade"`.
- Full result fidelity: text/JSON output and trust metadata (confidence,
  reasoning, sources, per-claim citations) preserved as native Python objects
  on `content.additional_properties["nimble"]`.
