"""Shared credential-free test fixtures.

Every test in this suite runs against a mocked HTTP transport
(``httpx.MockTransport``) wired *through the real ``nimble_python`` client*
-- so the typed-SDK request/response shape is exercised end to end, not a
hand-rolled fake. No test in this package ever makes a live network call or
requires ``NIMBLE_API_KEY``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from nimble_python import AsyncNimble

from nimble_agent_framework._client import build_async_client

AGENT_ID = "wsa_00000000000000000000000000"
RUN_ID = "task_run_00000000000000000000000000"


def json_response(status_code: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status_code, json=body, request=httpx.Request("GET", "http://test"))


def run_body(
    *,
    status: str = "queued",
    run_id: str = RUN_ID,
    agent_id: str = AGENT_ID,
    effort: str = "high",
    interaction_id: str = "interaction_1",
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": run_id,
        "created_at": "2026-08-04T00:00:00Z",
        "effort": effort,
        "interaction_id": interaction_id,
        "is_active": status in ("queued", "running"),
        "status": status,
        "web_search_agent_id": agent_id,
    }
    if error is not None:
        body["error"] = error
    return body


def result_body(
    *,
    output_type: str = "text",
    content: Any = "The answer, with a citation.[1]",
    run_id: str = RUN_ID,
    agent_id: str = AGENT_ID,
) -> dict[str, Any]:
    return {
        "output": {
            "type": output_type,
            "content": content,
            "trust": {
                "confidence": "high",
                "reasoning": "Multiple corroborating primary sources.",
                "sources": [
                    {
                        "type": "primary",
                        "url": "https://example.com/a",
                        "title": "Example A",
                        "source_category": "official",
                    }
                ],
                "claims": (
                    [
                        {
                            "callout": 1,
                            "confidence": "high",
                            "reasoning": "Directly stated.",
                            "citations": [
                                {
                                    "url": "https://example.com/a",
                                    "excerpts": ["the exact supporting sentence"],
                                    "source_type": "primary",
                                }
                            ],
                        }
                    ]
                    if output_type == "text"
                    else [
                        {
                            "path": "$.field",
                            "confidence": "high",
                            "reasoning": "Directly stated.",
                            "citations": [{"url": "https://example.com/a"}],
                        }
                    ]
                ),
            },
        },
        "run": run_body(status="completed", run_id=run_id, agent_id=agent_id),
    }


class RecordingTransport(httpx.MockTransport):
    """A MockTransport that records every request it handled."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []

        def recording_handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        super().__init__(recording_handler)


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[AsyncNimble, RecordingTransport]:
    transport = RecordingTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://sdk.nimbleway.com")
    client = build_async_client(
        api_key="sk-test-not-a-real-key",
        client_source="microsoft-agent-framework",
        http_client=http_client,
    )
    return client, transport


@pytest.fixture
def agent_id() -> str:
    return AGENT_ID


@pytest.fixture
def run_id() -> str:
    return RUN_ID


__all__ = [
    "AGENT_ID",
    "RUN_ID",
    "RecordingTransport",
    "json_response",
    "make_client",
    "result_body",
    "run_body",
]
