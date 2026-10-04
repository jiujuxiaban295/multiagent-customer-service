import asyncio
import hmac
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from app.auth import Principal, issue_token, verify_token
from app.config import Settings
from app.models import build_model, track_usage


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    user_id: str = Field(min_length=1, max_length=80, pattern=r'^[\w.@-]+$')


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=10000)
    conv_id: str | None = None
    user_id: str | None = None
    mode: Literal['v3'] = 'v3'


class CloseRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    outcome: Literal['unknown', 'resolved', 'unresolved'] = 'unknown'
    confirmation: str = Field(default='', max_length=4000)
    evidence_message_id: str = Field(default='', max_length=100)
    actual_steps: list[str] = Field(default_factory=list, max_length=20)
    environment: str = Field(default='', max_length=500)


class ChatResponse(BaseModel):
    """Preserve the existing /chat fields while exposing v3 traces and route evidence."""
    conv_id: str
    request_id: str
    response: str
    mode: Literal['v3'] = 'v3'
    success: bool
    intent: str
    intent_group: str = 'other'
    agent_type: str
    agent_types: list[str] = Field(default_factory=list)
    primary_agent: str = ''
    supporting_agents: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tool_traces: list[dict] = Field(default_factory=list)
    routing_reason: str = ''
    routing_confidence: float = 0.0
    escalated: bool
    latency_ms: float
    knowledge_used: bool = False
    entities: dict[str, list[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: dict[str, float] = Field(default_factory=dict)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved: list[dict] = Field(default_factory=list)
    timestamp: str
    extra: dict = Field(default_factory=dict)


def create_app(settings=None, services=None):
    settings = settings or Settings.from_env()
    registry = CollectorRegistry()
    completed = Counter('customer_requests_completed', 'Only completed requests', ['success'], registry=registry)
    latency = Histogram('customer_http_seconds', 'Full chat HTTP latency', registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        if services is not None:
            app.state.services = services
            yield
            return
        from app.retrieval import KnowledgeIndex
        from app.storage import SessionStore, CaseService
        from app.routing import Router
        from app.skills import SkillManager
        from app.runtime import V3Pipeline
        from app.tools import build_tools
        if not settings.dev_api_key or not settings.session_signing_key:
            raise RuntimeError('请先配置 DEV_API_KEY 和 SESSION_SIGNING_KEY；见 .env.example')
        http = httpx.AsyncClient(timeout=settings.retrieval_timeout)
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        try:
            await redis_client.ping()
            model = build_model(settings)
            index = KnowledgeIndex(settings, http, model)
            await index.initialize()
            store = SessionStore(settings.sqlite_path, redis_client, model)
            cases = CaseService(store, index, model)
            router = Router(http, settings.jev_api_key, settings.jev_model,
                            settings.jev_endpoint, settings.jev_timeout)
            skills = SkillManager(settings.skills_dir)
            skills.load()
            pipeline = V3Pipeline(model, router, build_tools(index, cases), skills,
                                  settings.model_call_limit, settings.request_timeout, knowledge=index)
            app.state.services = {'store': store, 'cases': cases, 'pipeline': pipeline,
                                  'skills': skills, 'index': index, 'redis': redis_client}
            yield
        finally:
            if hasattr(app.state, 'services'):
                app.state.services['store'].close()
            await redis_client.aclose()
            await http.aclose()

    app = FastAPI(title='多agent编排客服', version='3.0.0', lifespan=lifespan,
                  description='Jev + 规则路由，LangChain 主辅 Agent，当前会话和已解决案例。')
    app.state.traces = {}

    def principal(authorization, conv_id=None, user_id=None):
        p = verify_token(authorization or '', settings.session_signing_key)
        if (conv_id and conv_id != p.conv_id) or (user_id and user_id != p.owner):
            raise HTTPException(403, '会话凭证与请求身份不匹配')
        return p

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={'detail': '会话或证据不存在'})

    @app.exception_handler(PermissionError)
    async def forbidden(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, content={'detail': '无权访问该会话'})

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={'detail': str(exc)})

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'mode': 'v3', 'jev_configured': bool(settings.jev_api_key)}

    @app.post('/sessions', status_code=201)
    async def create_session(req: SessionRequest, x_dev_key: str = Header(default='')):
        if not settings.dev_api_key or not hmac.compare_digest(x_dev_key, settings.dev_api_key):
            raise HTTPException(401, '本地开发会话创建需要 X-Dev-Key')
        s = app.state.services['store'].create_session(req.user_id, settings.business_scope)
        token = issue_token(Principal(req.user_id, settings.business_scope, s['conv_id']), settings.session_signing_key)
        return {**s, 'session_token': token}

    @app.post('/chat', response_model=ChatResponse)
    async def chat(req: ChatRequest, authorization: str = Header(default='')):
        from app.contracts import PipelineInput
        if not req.message.strip():
            raise HTTPException(422, 'message 不能为空')
        p = principal(authorization, req.conv_id, req.user_id)
        store, pipeline = app.state.services['store'], app.state.services['pipeline']
        start = perf_counter()
        success = False
        try:
            async with store.lock(p.conv_id, p.owner, p.scope):
                snapshot = await store.snapshot(p.conv_id, p.owner, p.scope)
                result = await pipeline.run(PipelineInput(req.message, p.owner, p.conv_id, p.scope,
                    snapshot['summary'], snapshot['history']))
                # Exactly one owner of session writes; agents/middleware never append.
                with track_usage() as memory_usage:
                    await store.append_turn(p.conv_id, p.owner, p.scope, req.message,
                        result.response, result.request_id, result.tool_traces)
                result.extra['memory_usage'] = {'llm_calls': memory_usage.llm_calls,
                    'input_tokens': memory_usage.input_tokens, 'output_tokens': memory_usage.output_tokens}
                result.latency_ms = (perf_counter() - start) * 1000
                success = result.success
                data = asdict(result)
                role_results = result.extra.get('role_results', [])
                actual_agent = role_results[0]['role'] if len(role_results) == 1 else result.extra['primary_agent']
                data.update({'conv_id': p.conv_id, 'timestamp': datetime.now(timezone.utc).isoformat(),
                    'agent_type': actual_agent, 'intent': result.extra['intent'],
                    'primary_agent': result.extra['primary_agent'],
                    'agent_types': result.extra.get('agent_types', [result.extra['primary_agent']]),
                    'intent_group': result.extra['primary_agent'] if result.extra['intent'] != 'unknown' else 'other',
                    'routing_reason': result.extra['routing_reason'],
                    'supporting_agents': result.extra['supporting_agents'],
                    'knowledge_used': bool(result.retrieved), 'entities': result.extra.get('entities', {})})
                probability = result.extra.get('jev', {}).get('choice_probabilities', {}).get(result.extra['intent'], 0.0)
                data['intent_confidence'] = probability
                data['intent_source_scores'] = {'jev': probability} if probability else {}
                # Fusion scores are not calibrated probabilities; preserve the legacy field without fabricating one.
                data['routing_confidence'] = 0.0
                data['extra']['routing_confidence_status'] = 'unavailable; use routing_scores as uncalibrated scores'
                # ponytail: bounded in-process trace cache; use persistent trace store for multi-worker history.
                if len(app.state.traces) >= 500:
                    app.state.traces.pop(next(iter(app.state.traces)))
                app.state.traces[result.request_id] = (p, result.tool_traces)
                return data
        except TimeoutError:
            raise HTTPException(504, '请求处理超时或会话正在处理，请稍后重试') from None
        finally:
            completed.labels(success=str(success).lower()).inc()
            latency.observe(perf_counter() - start)

    @app.get('/sessions/{conv_id}')
    async def session(conv_id: str, authorization: str = Header(default='')):
        p = principal(authorization, conv_id)
        store = app.state.services['store']
        return {**store.get_session(conv_id, p.owner, p.scope),
                **await store.snapshot(conv_id, p.owner, p.scope)}

    @app.post('/sessions/{conv_id}/close')
    async def close(conv_id: str, req: CloseRequest, authorization: str = Header(default='')):
        p = principal(authorization, conv_id)
        store, cases = app.state.services['store'], app.state.services['cases']
        async with store.lock(conv_id, p.owner, p.scope, allow_closed=True):
            evidence = req.evidence_message_id
            # Confirmation and closure are serialized with chat as a single lifecycle operation.
            if req.confirmation and store.get_session(conv_id, p.owner, p.scope)['status'] == 'open':
                evidence = await store.record_confirmation(conv_id, p.owner, p.scope, req.confirmation)
            with track_usage() as usage:
                result = await cases.close(conv_id, p.owner, p.scope, req.outcome, evidence,
                    req.actual_steps, req.environment, confirmation_source='user')
        result['extraction_usage'] = {'llm_calls': usage.llm_calls,
            'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens}
        return result

    @app.post('/sessions/{conv_id}/cases/retry')
    async def retry(conv_id: str, authorization: str = Header(default='')):
        p = principal(authorization, conv_id)
        return await app.state.services['cases'].retry(conv_id, p.owner, p.scope)

    @app.post('/sessions/{conv_id}/reopen')
    async def reopen(conv_id: str, authorization: str = Header(default='')):
        p = principal(authorization, conv_id)
        return await app.state.services['cases'].reopen(conv_id, p.owner, p.scope)

    @app.post('/sessions/{conv_id}/cases/revoke')
    async def revoke(conv_id: str, authorization: str = Header(default='')):
        p = principal(authorization, conv_id)
        return await app.state.services['cases'].revoke(conv_id, p.owner, p.scope)

    @app.get('/trace/tool/{request_id}')
    async def trace(request_id: str, authorization: str = Header(default='')):
        p = principal(authorization)
        if request_id not in app.state.traces:
            raise HTTPException(404, '轨迹不存在或已过期')
        owner, traces = app.state.traces[request_id]
        if owner != p:
            raise HTTPException(403, '无权访问此轨迹')
        return {'request_id': request_id, 'tool_traces': traces}

    @app.get('/skills')
    async def skills(authorization: str = Header(default='')):
        principal(authorization)
        return app.state.services['skills'].summary()

    @app.post('/skills/reload')
    async def reload_skills(authorization: str = Header(default='')):
        principal(authorization)
        app.state.services['skills'].reload()
        return app.state.services['skills'].summary()

    @app.get('/metrics', response_class=PlainTextResponse)
    async def metrics():
        return PlainTextResponse(generate_latest(registry), media_type='text/plain; version=0.0.4')

    return app


app = create_app()
