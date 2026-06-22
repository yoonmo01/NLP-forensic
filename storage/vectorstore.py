"""벡터/하이브리드 검색 저장소.

VectorStore: 공통 인터페이스
LocalHybridStore: BM25(희소) + 밀집 코사인 → RRF 융합. 순수 파이썬, Windows OK (기본)
MilvusStore: pymilvus 백엔드(밀집 ANN + 키워드 부스트). Docker/WSL Milvus 필요 (옵션)

레코드 스키마: {entity_id, doc_id, entity_text, entity_type, context, vector?}
검색 결과: 위 필드 + score
"""
from __future__ import annotations
import json
import math
import os
import re
from abc import ABC, abstractmethod

import numpy as np

from config import CFG

_TOKEN = re.compile(r"[0-9a-zA-Z]+|[가-힣]+")


def tokenize(text: str) -> list[str]:
    """간단 토크나이저: 영숫자 런 + 한글 음절 런(+한글 2-gram). 한국어 BM25 보강용."""
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    grams: list[str] = []
    for t in toks:
        if re.match(r"[가-힣]+", t) and len(t) >= 2:
            grams += [t[i:i + 2] for i in range(len(t) - 1)]
    return toks + grams


def _cosine(mat: np.ndarray, q: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return np.array([])
    a = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    b = q / (np.linalg.norm(q) + 1e-9)
    return a @ b


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """여러 순위 리스트(인덱스)를 Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class VectorStore(ABC):
    @abstractmethod
    def add(self, records: list[dict]) -> None: ...
    @abstractmethod
    def search(self, query_text: str, query_vec: list[float], *, top_k: int = 5,
               type_filter: str | None = None) -> list[dict]: ...
    @abstractmethod
    def count(self) -> int: ...


class LocalHybridStore(VectorStore):
    def __init__(self, path: str | None = None):
        self.path = (path or CFG.db_path) + ".vec.json"
        self.records: list[dict] = []
        self._mat: np.ndarray = np.zeros((0, 0))
        self._bm25 = None
        self._load()

    # --- 영속화 ---
    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.records = json.load(f)
            self._rebuild()

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False)

    def _rebuild(self) -> None:
        vecs = [r["vector"] for r in self.records if r.get("vector")]
        self._mat = np.array(vecs, dtype=np.float32) if vecs else np.zeros((0, 0))
        try:
            from rank_bm25 import BM25Okapi
            corpus = [tokenize(f"{r['entity_text']} {r.get('context','')}") for r in self.records]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        except Exception:
            self._bm25 = None

    # --- 인터페이스 ---
    def add(self, records: list[dict]) -> None:
        self.records.extend(records)
        self._save()
        self._rebuild()

    def count(self) -> int:
        return len(self.records)

    def search(self, query_text, query_vec, *, top_k=5, type_filter=None):
        if not self.records:
            return []
        idxs = list(range(len(self.records)))
        if type_filter:
            idxs = [i for i in idxs if self.records[i]["entity_type"] == type_filter]
            if not idxs:
                return []

        # 밀집 순위
        dense_rank: list[int] = []
        if self._mat.size and query_vec:
            sims = _cosine(self._mat, np.array(query_vec, dtype=np.float32))
            dense_rank = sorted(idxs, key=lambda i: -float(sims[i]))

        # 희소(BM25) 순위
        sparse_rank: list[int] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query_text))
            sparse_rank = sorted(idxs, key=lambda i: -float(scores[i]))

        rankings = [r[:50] for r in (dense_rank, sparse_rank) if r]
        if not rankings:
            return []
        fused = _rrf(rankings)
        order = sorted(fused.keys(), key=lambda i: -fused[i])[:top_k]
        out = []
        for i in order:
            rec = {k: v for k, v in self.records[i].items() if k != "vector"}
            rec["score"] = round(fused[i], 5)
            out.append(rec)
        return out


class MilvusStore(VectorStore):
    """옵션 백엔드. 밀집 ANN + 키워드 부스트(간이 하이브리드)."""

    def __init__(self, uri: str | None = None, dim: int | None = None):
        from pymilvus import MilvusClient  # 지연 임포트
        self.client = MilvusClient(uri=uri or CFG.milvus_uri)
        self.COLL = CFG.milvus_collection   # 실험·반복별 분리 위해 런타임 값 사용
        self.dim = dim
        self._ensure()

    def _ensure(self):
        if self.dim and not self.client.has_collection(self.COLL):
            self.client.create_collection(self.COLL, dimension=self.dim, auto_id=True,
                                          metric_type="COSINE")

    def add(self, records: list[dict]) -> None:
        if not records:
            return
        if self.dim is None and records[0].get("vector"):
            self.dim = len(records[0]["vector"])
            self._ensure()
        data = [{
            "vector": r["vector"], "entity_id": r["entity_id"], "doc_id": r["doc_id"],
            "entity_text": r["entity_text"], "entity_type": r["entity_type"],
            "context": r.get("context", ""),
        } for r in records if r.get("vector")]
        if data:
            self.client.insert(self.COLL, data)

    def count(self) -> int:
        try:
            return self.client.get_collection_stats(self.COLL).get("row_count", 0)
        except Exception:
            return 0

    def search(self, query_text, query_vec, *, top_k=5, type_filter=None):
        if not self.client.has_collection(self.COLL):
            return []                       # 인덱싱된 개체 없음 → 빈 결과
        flt = f'entity_type == "{type_filter}"' if type_filter else None
        res = self.client.search(
            self.COLL, data=[query_vec], limit=max(top_k * 4, 20), filter=flt or "",
            output_fields=["entity_id", "doc_id", "entity_text", "entity_type", "context"],
        )[0]
        q = set(tokenize(query_text))
        out = []
        for hit in res:
            e = hit["entity"]
            kw = len(q & set(tokenize(f"{e['entity_text']} {e.get('context','')}")))
            out.append({**e, "score": round(float(hit["distance"]) + 0.05 * kw, 5)})
        out.sort(key=lambda r: -r["score"])
        return out[:top_k]


def get_store() -> VectorStore:
    if CFG.vector_backend == "milvus":
        return MilvusStore()
    return LocalHybridStore()
