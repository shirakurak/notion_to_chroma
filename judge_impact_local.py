#!/usr/bin/env python
import os
import json
import argparse
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel


# =========================
# 設定
# =========================

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "asia-northeast1")

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "notion_docs")

EMBEDDING_MODEL_NAME = "text-embedding-004"
LLM_MODEL_NAME = "gemini-1.5-pro"


# =========================
# 初期化
# =========================

if not GOOGLE_CLOUD_PROJECT:
    raise RuntimeError("環境変数 GOOGLE_CLOUD_PROJECT が設定されていません。")
if not VERTEX_AI_LOCATION:
    raise RuntimeError("環境変数 VERTEX_AI_LOCATION が設定されていません。")

vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=VERTEX_AI_LOCATION)

embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
llm_model = GenerativeModel(LLM_MODEL_NAME)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)


# =========================
# 埋め込み & 検索
# =========================

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    responses = embedding_model.get_embeddings(texts)
    return [e.values for e in responses]


def retrieve_context(query: str, k: int = 5) -> Dict[str, Any]:
    """
    クエリ文から埋め込みを作り、ChromaDB から類似チャンクを取得する。
    戻り値: collection.query の戻り値そのまま
    """
    embeddings = embed_texts([query])
    results = collection.query(
        query_embeddings=embeddings,
        n_results=k,
    )
    return results


def format_context(results: Dict[str, Any]) -> str:
    """
    Chroma からの検索結果を、LLM に渡しやすいテキストに整形する。
    """
    docs = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])

    lines = []
    for i, (doc, meta) in enumerate(zip(docs[0], metadatas[0])):
        header = f"[チャンク {i+1}]"
        title = meta.get("title") if isinstance(meta, dict) else None
        if title:
            header += f" (title: {title})"
        lines.append(header)
        lines.append(doc)
        lines.append("")

    return "\n".join(lines)


# =========================
# LLM 呼び出し & 判定
# =========================

def build_prompt(slack_message: str, context_text: str) -> str:
    """
    Vertex AI の LLM に渡すプロンプトを組み立てる。
    """
    prompt = f"""
あなたは、MC のデータ分析基盤の担当エンジニアです。
以下の情報をもとに、本番DBのマイグレーションが
データ分析基盤（DWH や BI レポート、バッチ処理など）に影響があるかどうかを判定してください。

- 判定は、Notion の影響確認ドキュメント（以下のコンテキスト）に基づいて行ってください。
- 必ず、影響の有無だけでなく「なぜそう判断したか（どのテーブル・どのルールに基づくか）」も説明してください。
- 影響があるか自信がない場合は "UNKNOWN" を使って構いません。

[マイグレーション通知メッセージ]
{slack_message}

[影響確認ドキュメントから取得した関連コンテキスト]
{context_text}

出力は、次の JSON 形式 **のみ** で返してください。
日本語のコメントや説明文を JSON の外に書いてはいけません。

{{
  "impact": "YES" または "NO" または "UNKNOWN",
  "reason": "なぜその判定になったのかを日本語で簡潔に。どのテーブル／どのルールに基づいたかも書く。"
}}
"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    LLM を呼び出し、テキスト（JSON文字列）を返す。
    """
    response = llm_model.generate_content(prompt)
    return response.text


def extract_json(text: str) -> Dict[str, Any]:
    """
    LLM の出力から JSON を抽出して dict にする。
    - コードブロック ```json ... ``` に包まれていても頑張って取り出す
    """
    txt = text.strip()

    # ```json ... ``` 形式に包まれている場合を剥がす
    if txt.startswith("```"):
        # 先頭の ```xxx を削る
        first_newline = txt.find("\n")
        if first_newline != -1:
            txt = txt[first_newline + 1 :]
        # 末尾の ``` を削る
        if txt.endswith("```"):
            txt = txt[:-3].strip()

    # 余分なテキストが前後に付く可能性もあるので、
    # 最初の { から最後の } までを抜き出して JSON とみなす
    first_brace = txt.find("{")
    last_brace = txt.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        txt = txt[first_brace : last_brace + 1]

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # どうしてもパースできない場合の保険
        return {
            "impact": "UNKNOWN",
            "reason": f"LLM 出力から JSON を正しくパースできませんでした。raw_output={text!r}",
        }


def judge_impact(slack_message: str, top_k: int = 5) -> Dict[str, Any]:
    """
    メイン関数:
      Slack メッセージを入力に、
      - ChromaDB から関連コンテキストを取得
      - Vertex AI LLM に投げて JSON 形式の判定をもらう
    """
    # 1. RAG 検索
    results = retrieve_context(slack_message, k=top_k)
    context_text = format_context(results)

    # 2. プロンプト作成
    prompt = build_prompt(slack_message, context_text)

    # 3. LLM 呼び出し
    raw_output = call_llm(prompt)

    # 4. JSON 抽出
    result = extract_json(raw_output)
    return result


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slackのマイグレーション投稿に対する影響判定 (Chroma + Vertex AI RAG)"
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        help="マイグレーション通知メッセージ（Slack 投稿相当）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="RAG で取得するコンテキスト数 (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.message:
        slack_msg = args.message
    else:
        print("マイグレーション通知メッセージを入力してください（終了するには Ctrl+D / Ctrl+Z）。")
        print("----")
        slack_msg = ""
        try:
            for line in iter(input, ""):
                slack_msg += line + "\n"
        except (EOFError, KeyboardInterrupt):
            pass
        slack_msg = slack_msg.strip()
        if not slack_msg:
            print("メッセージが空でした。終了します。")
            return

    print("\n[INPUT MESSAGE]")
    print(slack_msg)
    print("\n[RUNNING JUDGE...]\n")

    result = judge_impact(slack_msg, top_k=args.top_k)

    print("[RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
