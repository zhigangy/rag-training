from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
import json

# Milvus
from pymilvus import connections, Collection, utility

# Chroma
import chromadb

# Internal
from services.embedding_service import EmbeddingService
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG

logger = logging.getLogger(__name__)


class SearchService:
    """
    搜索服务类，负责向量数据库的连接和向量搜索功能
    支持 Milvus 和 Chroma 两种向量数据库
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.chroma_persist_dir = CHROMA_CONFIG["persist_directory"]
        self.search_results_dir = "04-search-results"
        os.makedirs(self.search_results_dir, exist_ok=True)
        os.makedirs(self.chroma_persist_dir, exist_ok=True)

    def get_providers(self) -> List[Dict[str, str]]:
        """获取支持的向量数据库列表"""
        return [
            {"id": VectorDBProvider.MILVUS.value, "name": "Milvus"},
            {"id": VectorDBProvider.CHROMA.value, "name": "Chroma"}
        ]

    def list_collections(self, provider: str = VectorDBProvider.MILVUS.value) -> List[Dict[str, Any]]:
        """
        获取指定向量数据库中的所有集合
        Args:
            provider: 向量数据库类型 ("milvus" 或 "chroma")
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=self.milvus_uri)
                collections = []
                for name in utility.list_collections():
                    try:
                        col = Collection(name)
                        collections.append({
                            "id": name,
                            "name": name,
                            "count": col.num_entities
                        })
                    except Exception as e:
                        logger.error(f"Error loading Milvus collection {name}: {e}")
                return collections
            except Exception as e:
                logger.error(f"Error listing Milvus collections: {e}")
                raise
            finally:
                connections.disconnect("default")

        elif provider == VectorDBProvider.CHROMA:
            try:
                client = chromadb.PersistentClient(path=self.chroma_persist_dir)
                collections = []
                for col in client.list_collections():
                    collections.append({
                        "id": col.name,
                        "name": col.name,
                        "count": col.count()
                    })
                return collections
            except Exception as e:
                logger.error(f"Error listing Chroma collections: {e}")
                raise
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def save_search_results(self, query: str, collection_id: str, results: List[Dict[str, Any]]) -> str:
        """保存搜索结果到 JSON 文件（与 provider 无关）"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_base = "".join(c for c in collection_id if c.isalnum() or c in "._-")
            filename = f"search_{collection_base}_{timestamp}.json"
            filepath = os.path.join(self.search_results_dir, filename)

            search_data = {
                "query": query,
                "collection_id": collection_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved search results to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving search results: {e}")
            raise

    def _search_milvus(
        self,
        query: str,
        collection_id: str,
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 20
    ) -> List[Dict[str, Any]]:
        """在 Milvus 中执行向量搜索"""
        try:
            logger.info(f"[Milvus] Connecting to {self.milvus_uri}")
            connections.connect(alias="default", uri=self.milvus_uri)

            collection = Collection(collection_id)
            collection.load()

            # 获取 embedding 配置
            sample_entity = collection.query(
                expr="id >= 0",
                output_fields=["embedding_provider", "embedding_model"],
                limit=1
            )
            if not sample_entity:
                raise ValueError(f"Collection {collection_id} is empty")

            provider = sample_entity[0]["embedding_provider"]
            model = sample_entity[0]["embedding_model"]
            logger.info(f"[Milvus] Using embedding config: provider={provider}, model={model}")

            query_embedding = self.embedding_service.create_single_embedding(query, provider=provider, model=model)

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=f"word_count >= {word_count_threshold}",
                output_fields=[
                    "content", "document_name", "chunk_id", "total_chunks",
                    "word_count", "page_number", "page_range",
                    "embedding_provider", "embedding_model", "embedding_timestamp"
                ]
            )

            processed = []
            for hits in results:
                for hit in hits:
                    if hit.score >= threshold:
                        processed.append({
                            "text": hit.entity.content,
                            "score": float(hit.score),
                            "metadata": {
                                "source": hit.entity.document_name,
                                "page": hit.entity.page_number,
                                "chunk": hit.entity.chunk_id,
                                "total_chunks": hit.entity.total_chunks,
                                "page_range": hit.entity.page_range,
                                "embedding_provider": hit.entity.embedding_provider,
                                "embedding_model": hit.entity.embedding_model,
                                "embedding_timestamp": hit.entity.embedding_timestamp
                            }
                        })
            return processed
        except Exception as e:
            logger.error(f"[Milvus] Search error: {e}")
            raise
        finally:
            connections.disconnect("default")

    def _search_chroma(
        self,
        query: str,
        collection_id: str,
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 20
    ) -> List[Dict[str, Any]]:
        """在 Chroma 中执行向量搜索"""
        try:
            client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            collection = client.get_collection(collection_id)

            # 从 collection metadata 获取 embedding 配置（需在 index 时存入）
            col_meta = collection.metadata or {}
            provider = col_meta.get("embedding_provider", "unknown")
            model = col_meta.get("embedding_model", "unknown")
            logger.info(f"[Chroma] Using embedding config from metadata: provider={provider}, model={model}")

            query_embedding = self.embedding_service.create_single_embedding(query, provider=provider, model=model)

            # Chroma 使用 where 过滤
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"word_count": {"$gte": word_count_threshold}}
            )

            processed = []
            ids_list = results.get("ids", [[]])[0]
            if not ids_list:
                return processed

            distances = results.get("distances", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i in range(len(ids_list)):
                # Chroma 默认使用 L2；若 collection 创建时指定 cosine，则 distance = 1 - cosine_sim
                cosine_sim = 1 - distances[i] if distances else 0.0
                if cosine_sim < threshold:
                    continue

                meta = metadatas[i] or {}
                processed.append({
                    "text": documents[i],
                    "score": float(cosine_sim),
                    "metadata": {
                        "source": meta.get("document_name", ""),
                        "page": meta.get("page_number", ""),
                        "chunk": meta.get("chunk_id", 0),
                        "total_chunks": meta.get("total_chunks", 0),
                        "page_range": meta.get("page_range", ""),
                        "embedding_provider": meta.get("embedding_provider", ""),
                        "embedding_model": meta.get("embedding_model", ""),
                        "embedding_timestamp": meta.get("embedding_timestamp", "")
                    }
                })
            return processed
        except Exception as e:
            logger.error(f"[Chroma] Search error: {e}")
            raise

    async def search(
        self,
        query: str,
        collection_id: str,
        provider: str,  # ← 显式传入 provider（推荐）
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 20,
        save_results: bool = False
    ) -> Dict[str, Any]:
        """
        执行向量搜索（支持 Milvus 和 Chroma）
        
        Args:
            query: 搜索文本
            collection_id: 集合名称
            provider: 向量数据库类型 ("milvus" 或 "chroma")
            top_k: 返回结果数
            threshold: 相似度阈值（cosine）
            word_count_threshold: 最小字数过滤
            save_results: 是否保存结果
            
        Returns:
            包含 results 的字典，可选 saved_filepath
        """
        logger.info(f"Search request - Provider: {provider}, Collection: {collection_id}, Query: {query}")

        if provider == VectorDBProvider.MILVUS:
            results = self._search_milvus(query, collection_id, top_k, threshold, word_count_threshold)
        elif provider == VectorDBProvider.CHROMA:
            results = self._search_chroma(query, collection_id, top_k, threshold, word_count_threshold)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        response_data = {"results": results}

        if save_results and results:
            try:
                filepath = self.save_search_results(query, collection_id, results)
                response_data["saved_filepath"] = filepath
            except Exception as e:
                logger.error(f"Failed to save search results: {e}")
                response_data["save_error"] = str(e)

        return response_data