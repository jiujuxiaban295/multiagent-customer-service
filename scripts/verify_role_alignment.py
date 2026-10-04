"""Real provider/API checks for restored role contracts, cooperation and handoff."""
import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from app.config import ROOT, Settings
from app.main import create_app
from app.profiles import PROFILES


SAMPLES = [
    ('technical', '这是合成联调样例。App 登录报 401，请调用 lookup_error_code 解释并给出排查步骤；无需查询政策或案例。'),
    ('compound', '这是合成联调样例。App 登录报 401，另有账单金额20元与10元需要比对。请技术角色调用 lookup_error_code，账单角色调用 compare_amounts；这里只做金额算术，无需查询政策或案例。'),
    ('escalation', '这是合成联调样例。请转人工，订单号 FIXTURE-1001。'),
]


async def main():
    settings = Settings.from_env()
    settings.business_scope = 'role-alignment-fixture-' + str(uuid4())
    report = {'data_kind': 'synthetic_fixture', 'model': settings.model,
        'verified_at': datetime.now(timezone.utc).isoformat(), 'samples': [],
        'role_model_parameters': {role: {'temperature': profile.temperature,
            'max_tokens': profile.max_tokens} for role, profile in PROFILES.items()
            if role != 'escalation'}}
    with tempfile.TemporaryDirectory(prefix='role-alignment-') as directory:
        settings.sqlite_path = str(Path(directory) / 'records.sqlite3')
        settings.chroma_path = str(Path(directory) / 'chroma')
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            store, created = app.state.services['store'], []
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                    base_url='http://fixture', timeout=180) as client:
                    for name, message in SAMPLES:
                        owner = 'fixture-' + name
                        response = await client.post('/sessions', json={'user_id': owner},
                            headers={'X-Dev-Key': settings.dev_api_key})
                        response.raise_for_status()
                        session = response.json()
                        created.append((session['conv_id'], owner))
                        response = await client.post('/chat', json={'message': message},
                            headers={'Authorization': 'Bearer ' + session['session_token']})
                        response.raise_for_status()
                        data = response.json()
                        assert data['success'], data['extra']['agent_results']
                        assert data['extra']['jev']['request_count'] == 1
                        assert len(store.raw_messages(session['conv_id'], owner, settings.business_scope)) == 2
                        if name == 'technical':
                            assert 'lookup_error_code' in data['tools_used'], data['response']
                        elif name == 'compound':
                            assert {'technical', 'billing'}.issubset(data['agent_types']), data['extra']['routing_scores']
                            assert {'lookup_error_code', 'compare_amounts'}.issubset(data['tools_used']), data['response']
                            assert len(data['extra']['role_results']) == 2
                        else:
                            assert data['llm_calls'] == 0 and data['escalated']
                            assert data['extra']['handoff_summaries'][0]['entities']['order_id'] == ['FIXTURE-1001']
                        item = {key: data[key] for key in ('primary_agent', 'supporting_agents',
                            'agent_types', 'tools_used', 'llm_calls', 'input_tokens', 'output_tokens',
                            'response', 'escalated')}
                        item.update(sample=name, jev_request_count=data['extra']['jev']['request_count'],
                            session_message_records=2, handoff_summaries=data['extra']['handoff_summaries'],
                            role_results=data['extra']['role_results'])
                        report['samples'].append(item)
                        print(json.dumps({'sample': name, 'success': True, 'agents': data['agent_types'],
                            'tools': data['tools_used'], 'llm_calls': data['llm_calls']}, ensure_ascii=False), flush=True)
            finally:
                for conv_id, owner in created:
                    await app.state.services['redis'].delete(store._key(conv_id, owner, settings.business_scope))
    report['success'] = True
    report['checks'] = ['real role model settings and tool rounds', 'primary-supporting composition',
        'deterministic handoff without generation', 'one Jev request per sample',
        'one persisted user-assistant pair per sample']
    (ROOT / 'docs/role-alignment-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
