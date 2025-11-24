#!/usr/bin/env python
import os
import argparse
from typing import List, Dict, Any

from notion_client import Client as NotionClient
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import vertexai
from vertexai.language_models import TextEmbeddingModel


# =========================
# Vertex AI Embedding
# =========================

class VertexAIEmbedder:
    def __init__(self, project_id: str, location: str, model_name: str = "text-embedding-004"):
        vertexai.init(project=project_id, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        texts の配列を Embedding ベクトルの配列に変換する。
        """
        if not texts:
            return []
        responses = self.model.get_embeddings(texts)
        return [e.values for e in responses]


# =========================
# Notion からテキスト抽出
# =========================

class NotionPageFetcher:
    def __init__(self, api_key: str):
        self.client = NotionClient(auth=api_key)

    def fetch_page(self, page_id: str) -> Dict[str, Any]:
        """
        指定ページIDのタイトルと全テキストを取得して返す。

        戻り値:
        {
          "page_id": ...,
          "title": ...,
          "content": "ページ全体のテキスト",
        }
        """
        page = self.client.pages.retrieve(page_id=page_id)

        # タイトル (プロパティ名 "Name" を想定。違う場合はここ調整)
        # title プロパティのキーはワークスペースによって違うので、柔軟にしたければ引数化してもよい
        title_prop = None
        for prop_name, prop_value in page["properties"].items():
            if prop_value["type"] == "title":
                title_prop = prop_value
                break

        if title_prop and title_prop["title"]:
            title = "".join([t["plain_text"] for t in title_prop["title"]])
        else:
            title = f"Untitled ({page_id})"

        # ブロック配下のテキストをすべて取得
        texts: List[str] = []
        self._collect_block_texts(page_id, texts)

        content = "\n".join(texts)
        return {
            "page_id": page_id,
            "title": title,
            "content": content,
        }

    def _collect_block_texts(self, block_id: str, texts: List[str]):
        """
        指定ブロック配下の rich_text を再帰的に集める。
        """
        cursor = None
        while True:
            resp = self.client.blocks.children.list(
                block_id=block_id,
                start_cursor=cursor,
            )
            for block in resp["results"]:
                block_type = block["type"]
                block_data = block.get(block_type, {})

                # rich_text を持つタイプからテキスト抽出
                rich_texts = block_data.get("rich_text", [])
                if rich_texts:
                    txt = "".join(rt.get("plain_text", "") for rt in rich_texts).strip()
                    if txt:
                        texts.append(txt)

                # 見出しのテキストを別で残したい場合はここで加工してもよい

                # 子ブロックがある場合は再帰
                if block.get("has_children"):
                    self._collect_block_texts(block["id"], texts)

            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")


# =========================
# ChromaDB へのインデックス
# =========================

class NotionChromaIndexer:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        embedder: VertexAIEmbedder,
    ):
        # ローカルの永続ストレージ用クライアント
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
            )
        )
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder

        # テキスト分割設定
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

    def index_page(self, page_record: Dict[str, Any]):
        """
        NotionPageFetcher.fetch_page の戻り値を受け取って、
        ChromaDB にチャンク＋Embedding として保存する。
        """
        page_id = page_record["page_id"]
        title = page_record["title"]
        content = page_record["content"]

        print(f"[INFO] Indexing page: {title} ({page_id})")

        if not content.strip():
            print("[WARN] Page content is empty. Skipping.")
            return

        # テキストチャンク化
        chunks = self.splitter.split_text(content)
        print(f"[INFO] Split into {len(chunks)} chunk(s).")

        # Embedding ベクトル
        embeddings = self.embedder.embed_texts(chunks)

        # ドキュメントID とメタデータ
        ids = [f"{page_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "page_id": page_id,
                "title": title,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # Chroma に upsert
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[INFO] Upserted {len(chunks)} chunks into collection '{self.collection.name}'.")


# =========================
# CLI エントリポイント
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index Notion pages into ChromaDB using Vertex AI embeddings."
    )
    parser.add_argument(
        "--page-id",
        action="append",
        required=True,
        help="Notion page ID. You can specify this option multiple times.",
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to ChromaDB persistent directory (default: ./chroma_db).",
    )
    parser.add_argument(
        "--collection",
        default="notion_docs",
        help="ChromaDB collection name (default: notion_docs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    notion_api_key = os.environ.get("NOTION_API_KEY")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_AI_LOCATION")

    if not notion_api_key:
        raise RuntimeError("NOTION_API_KEY is not set.")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set.")
    if not location:
        raise RuntimeError("VERTEX_AI_LOCATION is not set.")

    # 準備
    notion_fetcher = NotionPageFetcher(api_key=notion_api_key)
    embedder = VertexAIEmbedder(project_id=project_id, location=location)
    indexer = NotionChromaIndexer(
        db_path=args.db_path,
        collection_name=args.collection,
        embedder=embedder,
    )

    # 複数ページを順番にインデックス
    for page_id in args.page_id:
        page_record = notion_fetcher.fetch_page(page_id)
        indexer.index_page(page_record)


if __name__ == "__main__":
    main()
