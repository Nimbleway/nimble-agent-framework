"""Result-fidelity mapping: text/JSON output, trust, sources, claims, citations,
excerpts preserved as native structures, never lossy-stringified.
"""

from __future__ import annotations

from types import SimpleNamespace

from conftest import result_body
from nimble_python.types.agents.run_result_response import RunResultResponse
from pydantic import TypeAdapter

from nimble_agent_framework._mapping import result_to_agent_response

_ADAPTER: TypeAdapter[object] = TypeAdapter(RunResultResponse)


def _parse(body: dict) -> object:
    return _ADAPTER.validate_python(body)


def test_text_output_preserves_full_trust_structure_non_lossy() -> None:
    body = result_body(output_type="text", content="The sky is blue.[1]")
    result = _parse(body)

    response = result_to_agent_response(result, agent_id="framework-agent-id")

    assert response.agent_id == "framework-agent-id"
    assert response.response_id == body["run"]["id"]
    message = response.messages[-1]
    assert message.role == "assistant"
    assert message.text == "The sky is blue.[1]"

    nimble = message.contents[0].additional_properties["nimble"]
    assert nimble["output_type"] == "text"
    assert nimble["output"] == "The sky is blue.[1]"
    # Structured trust survives as native dict/list objects, not a string.
    trust = nimble["trust"]
    assert isinstance(trust, dict)
    assert trust["confidence"] == "high"
    claim = trust["claims"][0]
    assert claim["callout"] == 1
    citation = claim["citations"][0]
    assert citation["url"] == "https://example.com/a"
    assert citation["excerpts"] == ["the exact supporting sentence"]
    assert citation["source_type"] == "primary"
    source = trust["sources"][0]
    assert source["type"] == "primary"
    assert source["source_category"] == "official"

    run_envelope = nimble["run"]
    assert run_envelope["id"] == body["run"]["id"]
    assert run_envelope["web_search_agent_id"] == body["run"]["web_search_agent_id"]
    assert run_envelope["effort"] == body["run"]["effort"]
    assert run_envelope["interaction_id"] == body["run"]["interaction_id"]

    # No AgentResponse.value on a text-type result.
    assert response.value is None


def test_json_output_preserves_structured_value_non_lossy() -> None:
    structured = {"field": "value", "nested": {"a": [1, 2, 3]}}
    body = result_body(output_type="json", content=structured)
    result = _parse(body)

    response = result_to_agent_response(result, agent_id="framework-agent-id")

    message = response.messages[-1]
    nimble = message.contents[0].additional_properties["nimble"]
    assert nimble["output_type"] == "json"
    # The raw structured payload, untouched -- not a JSON string.
    assert nimble["output"] == structured
    assert isinstance(nimble["output"], dict)

    # AgentResponse.value is the framework's own field for exactly this.
    assert response.value == structured

    # A human-readable text rendering is still produced for chat surfaces.
    assert "field" in message.text
    assert "value" in message.text

    claim = nimble["trust"]["claims"][0]
    assert claim["path"] == "$.field"


def test_json_shaped_output_with_type_omitted_is_detected_as_json() -> None:
    """The published schema marks output `type` Optional -- a dict/list
    payload with no `type` must still take the JSON path, never be passed
    as non-string text."""

    structured = {"field": "value"}
    body = result_body(output_type="json", content=structured)
    del body["output"]["type"]
    result = _parse(body)

    response = result_to_agent_response(result, agent_id="framework-agent-id")

    message = response.messages[-1]
    nimble = message.contents[0].additional_properties["nimble"]
    assert nimble["output_type"] == "json"
    assert nimble["output"] == structured
    assert response.value == structured
    assert isinstance(message.text, str) and "field" in message.text


def test_json_shaped_output_declared_as_text_is_normalized_to_json() -> None:
    """A malformed explicit text discriminator must not put a dict in Content.text."""

    structured = {"field": "value"}
    result = SimpleNamespace(
        output=SimpleNamespace(
            type="text",
            content=structured,
            trust={"confidence": "high", "sources": [], "claims": []},
        ),
        run={"id": "task_run_test"},
    )

    response = result_to_agent_response(result, agent_id="framework-agent-id")

    message = response.messages[-1]
    nimble = message.contents[0].additional_properties["nimble"]
    assert nimble["output_type"] == "json"
    assert response.value == structured
    assert isinstance(message.text, str) and "field" in message.text


def test_text_output_with_type_omitted_is_detected_as_text() -> None:
    body = result_body(output_type="text", content="Plain answer.")
    del body["output"]["type"]
    result = _parse(body)

    response = result_to_agent_response(result, agent_id="framework-agent-id")
    nimble = response.messages[-1].contents[0].additional_properties["nimble"]
    assert nimble["output_type"] == "text"
    assert response.messages[-1].text == "Plain answer."
