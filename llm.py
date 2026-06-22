"""OpenAI 호환 LLM 래퍼: 채팅(JSON 모드), 임베딩, (선택) 비전.

전체 LLM 컴포넌트(NER, Text-to-SQL, 질의 분류, VLM 판단)를 단일 모델로 통일.
"""
from __future__ import annotations
import json
import time
from typing import Any

from openai import OpenAI

from config import CFG

_client: OpenAI | None = None
_embed_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"api_key": CFG.api_key}
        if CFG.base_url:
            kwargs["base_url"] = CFG.base_url
        _client = OpenAI(**kwargs)
    return _client


def embed_client() -> OpenAI:
    """임베딩 전용 클라이언트(채팅 게이트웨이와 분리). OpenAI/Upstage 등 호환 엔드포인트."""
    global _embed_client
    if _embed_client is None:
        kwargs: dict[str, Any] = {"api_key": CFG.embed_api_key}
        if CFG.embed_base_url:
            kwargs["base_url"] = CFG.embed_base_url
        _embed_client = OpenAI(**kwargs)
    return _embed_client


_vlm_client_obj: OpenAI | None = None


def vlm_client() -> OpenAI:
    """시각-언어(VLM) 전용 클라이언트. VLM_BASE_URL 미설정 시 메인 LLM 클라이언트 사용."""
    global _vlm_client_obj
    if _vlm_client_obj is None:
        kwargs: dict[str, Any] = {"api_key": CFG.api_key or "EMPTY"}
        kwargs["base_url"] = CFG.vlm_base_url or CFG.base_url
        _vlm_client_obj = OpenAI(**kwargs)
    return _vlm_client_obj


def vlm_model_name() -> str:
    """VLM 모델명. 미설정 시 메인 LLM 모델 사용."""
    return CFG.vlm_model or CFG.llm_model


def chat(system: str, user: str, *, temperature: float = 0.0, retries: int = 2,
         use_seed: bool = True) -> str:
    """일반 텍스트 응답. use_seed=False면 시드 미적용(self-consistency 표본 다양성 확보용)."""
    last = None
    params: dict[str, Any] = {
        "model": CFG.llm_model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if use_seed and CFG.seed is not None:   # 일관성: 고정 시드(지원 엔드포인트 한정)
        params["seed"] = CFG.seed
    for i in range(retries + 1):
        try:
            r = client().chat.completions.create(**params)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # 재시도 (rate limit/일시 오류)
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"LLM chat 실패: {last}")


def chat_json(system: str, user: str, *, temperature: float = 0.0, use_seed: bool = True) -> Any:
    """JSON 응답 강제 후 파싱. 실패 시 중괄호/대괄호 영역 추출 폴백."""
    sys2 = system + "\n반드시 유효한 JSON만 출력하라. 설명/마크다운 금지."
    txt = chat(sys2, user, temperature=temperature, use_seed=use_seed)
    return _loads(txt)


def _loads(txt: str) -> Any:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[txt.find("\n") + 1:] if "\n" in txt else txt
    try:
        return json.loads(txt)
    except Exception:
        # 첫 [ ... ] 또는 { ... } 영역만 시도
        for op, cl in (("[", "]"), ("{", "}")):
            i, j = txt.find(op), txt.rfind(cl)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(txt[i:j + 1])
                except Exception:
                    continue
        raise ValueError(f"JSON 파싱 실패: {txt[:200]}")


_local_embedder = None


def embed(texts: list[str], kind: str = "query") -> list[list[float]]:
    """밀집 임베딩. kind="passage"(문서 적재) | "query"(검색). 백엔드: openai | local | hash."""
    texts = [t if t.strip() else " " for t in texts]
    if not texts:
        return []
    backend = CFG.embed_backend
    if backend == "openai":
        model = CFG.embed_model_passage if kind == "passage" else CFG.embed_model
        r = embed_client().embeddings.create(model=model, input=texts)
        return [d.embedding for d in r.data]
    if backend == "local":
        global _local_embedder
        if _local_embedder is None:
            from sentence_transformers import SentenceTransformer  # 지연 임포트
            _local_embedder = SentenceTransformer(CFG.local_embed_model)
        return _local_embedder.encode(texts, normalize_embeddings=True).tolist()
    # hash: 오프라인 결정론 대체(의미 검색 품질 낮음, 동작 검증용)
    import hashlib
    dim = 256
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        out.append([((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)])
    return out
