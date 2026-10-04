"""One ChatModel and one usage owner for agents and all auxiliary model calls."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_anthropic import ChatAnthropic


@dataclass
class Usage:
    llm_calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_traces: list = field(default_factory=list)
    started: dict = field(default_factory=dict)


current_usage: ContextVar[Usage | None] = ContextVar('customer_usage', default=None)


@contextmanager
def track_usage():
    usage = Usage()
    token = current_usage.set(usage)
    try:
        yield usage
    finally:
        current_usage.reset(token)


class UsageCallback(AsyncCallbackHandler):
    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        if (usage := current_usage.get()) is not None:
            usage.llm_calls += 1
            usage.started[str(run_id)] = perf_counter()

    async def on_llm_end(self, response, *, run_id, **kwargs):
        if (usage := current_usage.get()) is not None:
            metadata = {}
            for batch in response.generations:
                for generation in batch:
                    metadata = getattr(getattr(generation, 'message', None), 'usage_metadata', None) or {}
                    usage.input_tokens += metadata.get('input_tokens', 0)
                    usage.output_tokens += metadata.get('output_tokens', 0)
            start = usage.started.pop(str(run_id), perf_counter())
            usage.model_traces.append({'run_id': str(run_id), 'success': True,
                'latency_ms': (perf_counter() - start) * 1000, **metadata})

    async def on_llm_error(self, error, *, run_id, **kwargs):
        if (usage := current_usage.get()) is not None:
            usage.failed_calls += 1
            start = usage.started.pop(str(run_id), perf_counter())
            usage.model_traces.append({'run_id': str(run_id), 'success': False,
                'latency_ms': (perf_counter() - start) * 1000, 'error': type(error).__name__})


def build_model(settings):
    if not settings.model_api_key:
        raise ValueError('ANTHROPIC_API_KEY 未配置')
    # Existing DeepSeek Anthropic-compatible integration, with thinking disabled (P13).
    return ChatAnthropic(model=settings.model, api_key=settings.model_api_key,
        base_url=settings.model_base_url, max_tokens=1600, temperature=0.2,
        timeout=60, max_retries=0, callbacks=[UsageCallback()],
        model_kwargs={'extra_body': {'thinking': {'type': 'disabled'}}})
