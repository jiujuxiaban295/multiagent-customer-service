"""Real services + explicitly synthetic, isolated API acceptance fixtures."""
import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import httpx
from app.config import Settings, ROOT
from app.main import create_app


async def main():
    settings = Settings.from_env()
    # Every generated conversation is labelled fixture and uses disposable storage.
    with tempfile.TemporaryDirectory(prefix='customer-v3-api-fixture-') as directory:
        settings.sqlite_path = str(Path(directory) / 'records.sqlite3')
        settings.chroma_path = str(Path(directory) / 'chroma')
        settings.business_scope = 'fixture-' + str(uuid4())
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture', timeout=180) as client:
                async def session(owner):
                    r = await client.post('/sessions',json={'user_id':owner},headers={'X-Dev-Key':settings.dev_api_key})
                    r.raise_for_status()
                    return r.json()['conv_id'], {'Authorization':'Bearer '+r.json()['session_token']}
                alice, alice_auth = await session('fixture-alice@example.com')
                bob, bob_auth = await session('fixture-bob')
                r = await client.post('/chat',json={'message':'App 登录报401，请只用错误码辅助工具解释，不需要政策知识库。'},headers=alice_auth)
                r.raise_for_status()
                chat = r.json()
                assert chat['success'] and 'lookup_error_code' in chat['tools_used'], chat
                assert chat['extra']['jev']['request_count'] == 1
                state = await client.get('/sessions/'+alice,headers=alice_auth)
                assert len(state.json()['history']) == 2
                mismatch = await client.get('/sessions/'+alice,headers=bob_auth)
                assert mismatch.status_code == 403
                close_body = {'outcome':'resolved','confirmation':'我已清理缓存并重新登录，App 已恢复正常，问题已解决',
                              'actual_steps':['清理缓存','重新登录'],'environment':'App'}
                closed = await client.post(f'/sessions/{alice}/close',json=close_body,headers=alice_auth)
                closed.raise_for_status()
                resolution = closed.json()
                assert resolution['publication_status'] == 'published', resolution
                repeated = await client.post(f'/sessions/{alice}/close',json=close_body,headers=alice_auth)
                assert repeated.json()['case_id'] == resolution['case_id']
                hits = await app.state.services['cases'].search('App 登录报401',settings.business_scope,error_code='401')
                assert len(hits) == 1, hits
                public = json.dumps(hits,ensure_ascii=False)
                assert 'fixture-alice@example.com' not in public and alice not in public
                refused = await client.post('/chat',json={'message':'继续'},headers=alice_auth)
                assert refused.status_code == 409
                unresolved = await client.post(f'/sessions/{bob}/close',json={'outcome':'unresolved'},headers=bob_auth)
                assert unresolved.json()['publication_status'] == 'rejected'
                reopened = await client.post(f'/sessions/{alice}/reopen',headers=alice_auth)
                reopened.raise_for_status()
                assert await app.state.services['cases'].search('App 登录报401',settings.business_scope) == []
                report = {'data_kind':'synthetic_fixture','model':settings.model,
                    'jev_model':settings.jev_model,'success':True,'chat_llm_calls':chat['llm_calls'],
                    'tools':chat['tools_used'],'tool_traces':chat['tool_traces'],
                    'jev_request_count':chat['extra']['jev']['request_count'],
                    'latency_ms':chat['latency_ms'],'case_status':resolution['publication_status'],
                    'checks':['real model-tool-model','real Jev once','Redis exactly two messages',
                        'API ownership isolation','explicit short resolution','stable case id',
                        'real bge-m3 case search','sanitized cross-user reusable content',
                        'closed chat refused','unresolved rejected','reopen revoked']}
                (ROOT/'docs/api-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                print(json.dumps(report,ensure_ascii=False,indent=2))
                # Remove our disposable Redis snapshots; never enumerate unrelated keys.
                store = app.state.services['store']
                await app.state.services['redis'].delete(
                    store._key(alice,'fixture-alice@example.com',settings.business_scope),
                    store._key(bob,'fixture-bob',settings.business_scope))


if __name__ == '__main__':
    asyncio.run(main())
