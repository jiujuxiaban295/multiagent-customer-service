"""Real authorized DeepSeek rewrite/rerank, followed by an isolated full API check."""
import asyncio
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo
import httpx
from app.config import Settings, ROOT
from app.main import create_app
from app.models import build_model, track_usage
from app.retrieval import KnowledgeIndex, _message_text


class Recorder:
    def __init__(self, model):
        self.model = model
        self.calls = []

    async def ainvoke(self, messages):
        start = perf_counter()
        reply = await self.model.ainvoke(messages)
        self.calls.append({'messages': messages, 'output': _message_text(reply),
            'latency_ms': round((perf_counter() - start) * 1000, 2),
            'usage': reply.usage_metadata, 'model': reply.response_metadata.get('model')})
        return reply


async def main():
    settings = Settings.from_env()
    settings.rag_cache_ttl = 0
    model = Recorder(build_model(settings))
    report = {'timestamp': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        'provider': 'DeepSeek', 'model': settings.model,
        'embedding_model': settings.embedding_model,
        'knowledge_collection': settings.knowledge_collection,
        'cache_ttl': 0, 'verification_scope': 'real rewrite, parallel recall, deduplication, rerank and API answer',
        'queries': [], 'credentials_saved': False}
    api_only = '--api-only' in sys.argv
    if api_only:
        report = json.loads((ROOT / 'docs/rag-verification.json').read_text())
    direct_queries = [] if api_only else [
        ('退款审核通过后，微信和银行卡各需要多久到账？', 5),
        ('订单原价200元，使用20元优惠券，退货后可以退多少钱？', 3)]
    async with httpx.AsyncClient(timeout=60) as http:
        index = KnowledgeIndex(settings, http, model)
        if direct_queries:
            await index.initialize()
            report['chunks'] = await asyncio.to_thread(index.knowledge.count)
        for query, top_k in direct_queries:
            model.calls = []
            start = perf_counter()
            with track_usage() as usage:
                result = await index.search_knowledge(query, top_k=top_k)
            assert result['success'] and result['reranked'] and not result.get('degraded'), result
            assert usage.llm_calls == 2 and usage.failed_calls == 0, usage
            variants = json.loads(model.calls[0]['output'].strip())
            assert len(set([query, *variants])) == 4, variants
            order = json.loads(model.calls[1]['output'].strip())
            row = {'query': query, 'top_k': top_k, 'success': True, 'reranked': True, 'degraded': False,
                'latency_ms': round((perf_counter() - start) * 1000, 2),
                'rewrite_queries': [query, *variants], 'rerank_order': order,
                'candidate_count': len(json.loads(model.calls[1]['messages'][1][1])['candidates']),
                'results': result['results'], 'model_calls': model.calls,
                'usage': {'llm_calls': usage.llm_calls, 'failed_calls': usage.failed_calls,
                          'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens}}
            report['queries'].append(row)
            print(json.dumps({'query': query, 'reranked': True,
                'calls': usage.llm_calls, 'top_titles': [r['title'] for r in result['results']]}, ensure_ascii=False), flush=True)
        (ROOT / 'docs/rag-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # Full FastAPI chat uses actual router, agent and tool. Private messages stay in a disposable DB.
    with tempfile.TemporaryDirectory(prefix='customer-v3-rag-') as directory:
        settings.sqlite_path = str(Path(directory) / 'records.sqlite3')
        settings.business_scope = 'fixture-rag-' + str(uuid4())
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            recording_index_model = Recorder(app.state.services['index'].model)
            app.state.services['index'].model = recording_index_model
            search_results = []
            actual_search = app.state.services['index'].search_knowledge

            async def tracked_search(query, top_k=5):
                result = await actual_search(query, top_k)
                search_results.append({'query': query, 'top_k': top_k,
                    'success': result['success'], 'reranked': result['reranked'],
                    'degraded': result.get('degraded', False)})
                return result

            app.state.services['index'].search_knowledge = tracked_search
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture', timeout=180) as client:
                owner = 'fixture-rag-user'
                created = await client.post('/sessions', json={'user_id': owner}, headers={'X-Dev-Key': settings.dev_api_key})
                created.raise_for_status()
                conv_id = created.json()['conv_id']
                auth = {'Authorization': 'Bearer ' + created.json()['session_token']}
                question = '退款审核通过后，微信和银行卡各需要多久到账？请先检索正式知识库再回答。'
                response = await client.post('/chat', json={'message': question}, headers=auth)
                response.raise_for_status()
                result = response.json()
                assert result['success'] and result['knowledge_used'], result
                assert 'search_knowledge_base' in result['tools_used'], result
                assert result['extra']['jev']['request_count'] == 1
                assert search_results and all(r['success'] and not r['degraded'] for r in search_results), search_results
                assert any(r['reranked'] for r in search_results), search_results
                rewrite_count = sum(c['messages'][0][1].startswith('将问题改写') for c in recording_index_model.calls)
                rerank_count = sum(c['messages'][0][1].startswith('按与用户问题') for c in recording_index_model.calls)
                assert rewrite_count == len(search_results) and rerank_count >= 1
                assert any(r['title'] == '退款到账说明' for r in result['retrieved'])
                # Reviewed policy gives one 5-7 working-day window, without channel-specific windows.
                answer = re.sub(r'[\s*]', '', result['response'])
                assert re.search(r'5(?:[-–—~～]|至|到)7个?工作日', answer), result['response']
                assert '我帮你核对具体订单' not in answer and '我帮你查询订单状态' not in answer, result['response']
                report['api'] = {'data_kind': 'synthetic_fixture', 'question': question,
                    'success': result['success'], 'response': result['response'],
                    'knowledge_used': result['knowledge_used'], 'tools_used': result['tools_used'],
                    'tool_traces': result['tool_traces'], 'llm_calls': result['llm_calls'],
                    'input_tokens': result['input_tokens'], 'output_tokens': result['output_tokens'],
                    'latency_ms': result['latency_ms'], 'jev_request_count': 1,
                    'knowledge_searches': search_results, 'rewrite_calls': rewrite_count,
                    'rerank_calls': rerank_count,
                    'index_model_calls': recording_index_model.calls}
                store = app.state.services['store']
                await app.state.services['redis'].delete(store._key(conv_id, owner, settings.business_scope))
                print(json.dumps({'api_success': True, 'llm_calls': result['llm_calls'],
                    'knowledge_used': True, 'response': result['response']}, ensure_ascii=False), flush=True)
    report['success'] = True
    (ROOT / 'docs/rag-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Saved docs/rag-verification.json', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
