"""Import the reviewed Markdown KB into the new project's isolated Chroma storage."""

import argparse
import asyncio
from pathlib import Path

import httpx

from app.config import Settings
from app.models import build_model
from app.retrieval import KnowledgeIndex, parse_markdown_docs


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="data/kb/电商客服知识库.md")
    args = parser.parse_args()
    text = Path(args.path).read_text(encoding="utf-8")
    settings = Settings.from_env()
    async with httpx.AsyncClient() as client:
        index = KnowledgeIndex(settings, client, build_model(settings))
        await index.initialize()
        chunks = await index.import_documents(text)
    print(f"Imported {len(parse_markdown_docs(text))} documents / {chunks} chunks.")


if __name__ == "__main__":
    asyncio.run(main())
