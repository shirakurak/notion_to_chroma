# notion_to_chroma

## 準備

```sh
export NOTION_API_KEY="secret_xxx" # Notionのインテグレーションのシークレット
export GOOGLE_CLOUD_PROJECT="your-project-id"
export VERTEX_AI_LOCATION="asia-northeast1" # 例: 東京リージョン
# ChromaDB のパスとコレクション名（必要に応じて変更）
export CHROMA_DB_PATH="./chroma_db"
export CHROMA_COLLECTION="notion_docs"
```

```sh
pip install \
  notion-client \
  chromadb \
  langchain \
  google-cloud-aiplatform \
  vertexai
```

## 手順

1. `chroma_db` ディレクトリを GCS にアップロードする
2. aaa
3. bbb
