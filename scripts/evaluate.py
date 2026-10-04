"""API-level v3 evaluation; each item has its own credential and session."""
import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
import httpx
from app.config import Settings


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', type=Path, help='JSON list: {id,messages:[...],must_include:[...]}')
    parser.add_argument('--url', default='http://127.0.0.1:8010')
    parser.add_argument('--out', type=Path, default=Path('data/eval/latest.json'))
    args = parser.parse_args()
    settings = Settings.from_env()
    cases = json.loads(args.cases.read_text())
    results = []
    async with httpx.AsyncClient(base_url=args.url, timeout=180) as client:
        for item in cases:
            created = await client.post('/sessions', json={'user_id': 'eval-' + str(item['id'])},
                headers={'X-Dev-Key': settings.dev_api_key})
            created.raise_for_status()
            auth = {'Authorization': 'Bearer ' + created.json()['session_token']}
            turns = []
            for message in item['messages']:
                response = await client.post('/chat', json={'message': message}, headers=auth)
                response.raise_for_status()
                turns.append(response.json())
            final = turns[-1]
            # Deterministic checklist, never presented as a real resolution rate or LLM quality score.
            checks = {text: text in final['response'] for text in item.get('must_include', [])}
            results.append({'id': item['id'], 'turns': turns, 'checks': checks})
    report = {'mode': 'v3', 'cases': results, 'successful_execution_rate':
        mean(all(t['success'] for t in row['turns']) for row in results) if results else 0,
        'notes': '执行成功率不是实际解决率。关键词检查仅供人工审阅辅助。'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'已保存 {len(results)} 条 API 评测记录：{args.out}')


if __name__ == '__main__':
    asyncio.run(main())
