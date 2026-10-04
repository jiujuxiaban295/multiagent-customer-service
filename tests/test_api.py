import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
import httpx
from app.config import Settings
from app.contracts import PipelineResult
from app.main import create_app
from app.skills import SkillManager
from app.storage import SessionStore, CaseService
from tests.test_storage import FakeRedis, FakeIndex


class Pipeline:
    def __init__(self, cases):
        self.cases = cases
        self.inputs = []

    async def run(self, inp):
        self.inputs.append(inp)
        await asyncio.sleep(0.001)
        response = '请清理缓存再重新启动 App。'
        if inp.message == '查询相似已解决案例':
            response = json.dumps(await self.cases.search(inp.message, inp.scope), ensure_ascii=False)
        return PipelineResult(request_id=str(uuid4()), response=response, success=True,
            extra={'primary_agent': 'technical', 'supporting_agents': [],
                   'intent': 'technical', 'routing_reason': 'fixture'})


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp.name) / 'records.sqlite3', FakeRedis())
        self.index = FakeIndex()
        self.cases = CaseService(self.store, self.index)
        self.pipeline = Pipeline(self.cases)
        self.settings = Settings(dev_api_key='dev-fixture-key', session_signing_key='signed-fixture-key')
        self.app = create_app(self.settings, {'store': self.store, 'cases': self.cases,
            'pipeline': self.pipeline, 'skills': SkillManager(self.settings.skills_dir)})
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.store.close()
        self.temp.cleanup()

    async def session(self, owner='alice'):
        response = await self.client.post('/sessions', json={'user_id': owner},
            headers={'X-Dev-Key': self.settings.dev_api_key})
        self.assertEqual(response.status_code, 201)
        data = response.json()
        return data['conv_id'], {'Authorization': 'Bearer ' + data['session_token']}

    async def test_auth_ownership_and_mode_boundary(self):
        self.assertEqual((await self.client.post('/sessions', json={'user_id': 'alice'})).status_code, 401)
        conv, auth = await self.session()
        bob, bob_auth = await self.session('bob')
        self.assertEqual((await self.client.post('/chat', json={'message':'hello'})).status_code, 401)
        self.assertEqual((await self.client.post('/chat', json={'message':'hello','user_id':'bob'}, headers=auth)).status_code, 403)
        self.assertEqual((await self.client.get('/sessions/'+conv, headers=bob_auth)).status_code, 403)
        self.assertEqual((await self.client.post('/chat', json={'message':'hello','mode':'v2'}, headers=auth)).status_code, 422)
        self.assertEqual((await self.client.post('/chat', json={'message':'  '}, headers=auth)).status_code, 422)

    async def test_chat_once_history_concurrency_and_trace_ownership(self):
        conv, auth = await self.session()
        replies = await asyncio.gather(*[self.client.post('/chat', json={'message': str(i)}, headers=auth) for i in range(2)])
        self.assertTrue(all(r.status_code == 200 for r in replies))
        self.assertTrue({'intent_group', 'agent_types', 'primary_agent', 'routing_confidence',
            'knowledge_used', 'entities', 'intent_confidence', 'intent_source_scores'}.issubset(replies[0].json()))
        self.assertIn('unavailable', replies[0].json()['extra']['routing_confidence_status'])
        records = self.store.raw_messages(conv, 'alice', self.settings.business_scope)
        self.assertEqual(len(records), 4)
        self.assertEqual(len(self.pipeline.inputs[0].history), 0)
        self.assertEqual(len(self.pipeline.inputs[1].history), 2)
        rid = replies[0].json()['request_id']
        bob, bob_auth = await self.session('bob')
        self.assertEqual((await self.client.get('/trace/tool/'+rid, headers=bob_auth)).status_code, 403)
        self.assertEqual((await self.client.get('/trace/tool/'+rid, headers=auth)).status_code, 200)

    async def test_api_short_resolution_shared_case_closed_and_reopen(self):
        conv, auth = await self.session('alice@example.com')
        await self.client.post('/chat', json={'message':'App 登录报 401，订单 A10050，手机号 13812345678'}, headers=auth)
        body = {'outcome':'resolved', 'confirmation':'我已清理缓存并重新启动 App，问题已解决',
                'actual_steps':['清理缓存','重新启动 App'], 'environment':'App'}
        response = await self.client.post(f'/sessions/{conv}/close',json=body,headers=auth)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['publication_status'],'published')
        repeated = await self.client.post(f'/sessions/{conv}/close',json=body,headers=auth)
        self.assertEqual(response.json()['case_id'], repeated.json()['case_id'])
        self.assertEqual(self.index.upserts,1)
        self.assertEqual((await self.client.post('/chat',json={'message':'more'},headers=auth)).status_code,409)
        bob, bob_auth = await self.session('bob')
        reused = await self.client.post('/chat',json={'message':'查询相似已解决案例'},headers=bob_auth)
        public = reused.json()['response']
        self.assertIn(response.json()['case_id'],public)
        for private in ['alice@example.com','13812345678','A10050',conv]:
            self.assertNotIn(private,public)
        self.assertEqual((await self.client.post(f'/sessions/{conv}/reopen',headers=auth)).status_code,200)
        self.assertEqual(await self.cases.search('App',self.settings.business_scope),[])

    async def test_pending_index_retry_and_unresolved_never_published(self):
        conv, auth = await self.session()
        await self.client.post('/chat',json={'message':'App 启动失败'},headers=auth)
        self.index.fail_write=True
        r=await self.client.post(f'/sessions/{conv}/close',json={'outcome':'resolved',
            'confirmation':'我已清理缓存，问题已解决','actual_steps':['清理缓存']},headers=auth)
        self.assertEqual(r.json()['publication_status'],'pending')
        self.index.fail_write=False
        r=await self.client.post(f'/sessions/{conv}/cases/retry',headers=auth)
        self.assertEqual(r.json()['publication_status'],'published')
        other, other_auth=await self.session('bob')
        r=await self.client.post(f'/sessions/{other}/close',json={'outcome':'unresolved'},headers=other_auth)
        self.assertEqual(r.json()['publication_status'],'rejected')

    async def test_actual_agent_after_fallback_is_distinct_from_selected_primary(self):
        from langchain_core.messages import AIMessage
        from app.runtime import V3Pipeline
        from app.tools import build_tools
        from tests.test_runtime import ScriptModel, FixedRouter, FakeSkills, FakeIndex as KnowledgeFixture

        model = ScriptModel(behavior=lambda event: RuntimeError('fixture technical failure')
            if event['role'] == 'technical' else AIMessage(content='通用客服接续处理。'))
        self.app.state.services['pipeline'] = V3Pipeline(model, FixedRouter('technical'),
            build_tools(KnowledgeFixture(), self.cases), FakeSkills(), timeout=5)
        conv, auth = await self.session()
        response = await self.client.post('/chat', json={'message': 'fixture technical issue'}, headers=auth)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['primary_agent'], 'technical')
        self.assertEqual(data['agent_type'], 'general')
        self.assertEqual(data['extra']['role_results'][0]['requested_role'], 'technical')
        self.assertEqual(len(self.store.raw_messages(conv, 'alice', self.settings.business_scope)), 2)
