"""Non-lossy mapping from Nimble Agent API V2 results to Agent Framework types.

The completed run's ``output`` (``type: "text"`` or ``type: "json"``) and
``trust`` (confidence, reasoning, sources, claims, citations, excerpts) are
attached verbatim as native Python objects on
``Content.additional_properties["nimble"]`` -- never stringified -- so a
consumer that wants the raw structured data never has to re-parse text that
was flattened for display.
"""

from __future__ import annotations

import json
from typing import Any

from agent_framework import AgentResponse, Content, Message

_STATE_KEY = "nimble"


def _model_dict(model: Any) -> dict[str, Any]:
    """Convert a nimble-python (Pydantic) response model to a plain dict.

    Returns native Python objects (dict/list/str/...), never a stringified
    blob, so trust/citation data downstream stays structured.
    """

    if hasattr(model, "model_dump"):
        value = model.model_dump(mode="json")
    elif hasattr(model, "to_dict"):
        value = model.to_dict()
    elif hasattr(model, "dict"):
        value = model.dict()
    elif isinstance(model, dict):
        value = model
    else:
        raise TypeError(f"Unsupported Nimble SDK response type: {type(model)!r}")
    if not isinstance(value, dict):
        raise TypeError("Nimble SDK response conversion did not return a dict")
    return value


def result_to_agent_response(result: Any, *, agent_id: str | None = None) -> AgentResponse:
    """Map a completed ``TaskRunResultPublicV2`` (``result.output`` + ``result.run``)
    to an :class:`agent_framework.AgentResponse`.

    ``agent_id`` here is the *framework* agent's own id (``BaseAgent.id``),
    deliberately distinct from Nimble's ``web_search_agent_id`` -- callers
    pass ``self.id``. ``response_id`` is set to the Nimble run id, which is
    exactly what that field is for: the id of *this* response.
    """

    output = result.output
    # The published schema marks output `type` as Optional -- the server may
    # omit it even for structured output. Fall back to the content's actual
    # shape rather than assuming "text", so a dict/list payload is never
    # mis-rendered through the text path.
    output_type = getattr(output, "type", None)
    if output_type != "json" and isinstance(output.content, dict | list):
        output_type = "json"
    elif output_type is None:
        output_type = "text"
    trust_dict = _model_dict(output.trust)
    run_dict = _model_dict(result.run)

    structured_value: Any | None = None
    if output_type == "json":
        text = json.dumps(output.content, indent=2, default=str, sort_keys=True)
        structured_value = output.content
    else:
        text = output.content

    content = Content.from_text(
        text,
        additional_properties={
            _STATE_KEY: {
                "output_type": output_type,
                "output": output.content,
                "trust": trust_dict,
                "run": run_dict,
            }
        },
    )
    message = Message(role="assistant", contents=[content])
    return AgentResponse(
        messages=[message],
        response_id=run_dict.get("id"),
        agent_id=agent_id,
        value=structured_value,
    )
