from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineInput:
    message: str
    user_id: str
    conv_id: str
    scope: str
    context: str = ''
    history: list[dict[str, str]] | None = None


@dataclass
class PipelineResult:
    mode: str = 'v3'
    request_id: str = ''
    response: str = ''
    success: bool = False
    escalated: bool = False
    tools_used: list[str] = field(default_factory=list)
    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    inp: PipelineInput
    request_id: str
    route: Any
    role: str
    allowed_tools: frozenset[str]
    tool_traces: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)
    escalated: bool = False
    model_calls: int = 0
