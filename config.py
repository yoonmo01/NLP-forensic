"""중앙 설정 로더. .env 를 읽어 전역 설정 객체를 제공한다."""
from __future__ import annotations
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv 미설치 시에도 환경변수로 동작
    pass


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@dataclass
class Config:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    llm_model: str = os.getenv("LLM_MODEL", "gpt-5-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "embedding-query")
    embed_model_passage: str = os.getenv("EMBED_MODEL_PASSAGE", "embedding-passage")
    # 임베딩 백엔드: openai(별도 OpenAI/Upstage 키) | local(sentence-transformers) | hash(오프라인 대체)
    embed_backend: str = os.getenv("EMBED_BACKEND", "local")
    local_embed_model: str = os.getenv("LOCAL_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    # 임베딩 전용 키·엔드포인트(채팅 게이트웨이와 분리). 미설정 시 메인 키 사용.
    # OpenAI: base_url 비움 / Upstage: https://api.upstage.ai/v1
    embed_api_key: str = os.getenv("EMBED_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    embed_base_url: str | None = os.getenv("EMBED_BASE_URL") or None

    # Upstage Document Parse / Groundedness Check
    upstage_api_key: str = (
        os.getenv("UPSTAGE_API_KEY", "")
        if os.getenv("UPSTAGE_API_KEY", "") not in ("", "YOUR_UPSTAGE_KEY")
        else (os.getenv("EMBED_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""))
    )
    upstage_base_url: str = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
    upstage_document_endpoint: str = os.getenv(
        "UPSTAGE_DOCUMENT_PARSE_ENDPOINT",
        "https://api.upstage.ai/v1/document-digitization",
    )
    upstage_document_model: str = os.getenv("UPSTAGE_DOCUMENT_PARSE_MODEL", "document-parse")
    upstage_document_output_format: str = os.getenv("UPSTAGE_DOCUMENT_OUTPUT_FORMAT", "markdown")
    upstage_document_ocr: str = os.getenv("UPSTAGE_DOCUMENT_OCR", "auto")
    upstage_groundedness_model: str = os.getenv("UPSTAGE_GROUNDEDNESS_MODEL", "groundedness-check")
    upstage_timeout_sec: int = int(os.getenv("UPSTAGE_TIMEOUT_SEC", "300"))

    # NAVER Cloud CLOVA Speech / Object Storage
    ncloud_access_key: str = os.getenv("NCLOUD_ACCESS_KEY", "")
    ncloud_secret_key: str = os.getenv("NCLOUD_SECRET_KEY", "")
    ncloud_object_storage_endpoint: str = os.getenv(
        "NCLOUD_OBJECT_STORAGE_ENDPOINT",
        "https://kr.object.ncloudstorage.com",
    )
    ncloud_region: str = os.getenv("NCLOUD_REGION", "kr-standard")
    ncloud_bucket_name: str = os.getenv("NCLOUD_BUCKET_NAME", "")
    ncloud_clova_input_prefix: str = os.getenv("NCLOUD_CLOVA_INPUT_PREFIX", "original-mp4")
    ncloud_clova_output_prefix: str = os.getenv("NCLOUD_CLOVA_OUTPUT_PREFIX", "result-stt")
    # 두 가지 표기 모두 허용 (CLOVA_INVOKE_URL / CLOVA_SPEECH_INVOKE_URL)
    clova_invoke_url: str = os.getenv("CLOVA_INVOKE_URL", "") or os.getenv("CLOVA_SPEECH_INVOKE_URL", "")
    clova_secret_key: str = os.getenv("CLOVA_SECRET_KEY", "") or os.getenv("CLOVA_SPEECH_SECRET_KEY", "")
    # Object Storage 경로 사용 여부(기본 off → 파일 직접 업로드). 권한 이슈 회피.
    clova_use_object_storage: bool = os.getenv("CLOVA_USE_OBJECT_STORAGE", "").lower() in ("1", "true", "yes")
    clova_language: str = os.getenv("CLOVA_LANGUAGE", "ko-KR")
    clova_completion_mode: str = os.getenv("CLOVA_COMPLETION_MODE", "sync")
    clova_timeout_sec: int = int(os.getenv("CLOVA_TIMEOUT_SEC", "600"))

    # ===== 로컬 백엔드 (전 구성요소 로컬 전환) =====
    # VLM(이미지 캡션/OCR fallback): 별도 멀티모달 엔드포인트(미설정 시 메인 LLM 사용)
    vlm_base_url: str | None = os.getenv("VLM_BASE_URL") or None
    vlm_model: str = os.getenv("VLM_MODEL", "")
    # STT: faster_whisper(로컬, 인프로세스, 기본) | clova(레거시)
    stt_backend: str = os.getenv("STT_BACKEND", "faster_whisper")
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    stt_device: str = os.getenv("STT_DEVICE", "cuda")
    stt_compute_type: str = os.getenv("STT_COMPUTE_TYPE", "float16")
    # OCR: vlm(메인 VLM 재사용·기본·견고) | surya | paddleocr
    ocr_backend: str = os.getenv("OCR_BACKEND", "vlm")
    # 근거 검증: local_llm(로컬 LLM 판정·기본) | upstage(레거시) | off
    groundedness_backend: str = os.getenv("GROUNDEDNESS_BACKEND", "local_llm")

    # 일관성: 고정 시드(빈 값이면 비활성 — 미지원 엔드포인트 대비). self-consistency 표본 수.
    seed: int | None = (int(os.getenv("LLM_SEED")) if os.getenv("LLM_SEED") else None)
    self_consistency_n: int = int(os.getenv("SELF_CONSISTENCY_N", "1"))

    db_path: str = os.getenv("DB_PATH", "./forensic.db")
    relational_backend: str = os.getenv("RELATIONAL_BACKEND", "sqlite")  # sqlite | postgres
    database_url: str = os.getenv("DATABASE_URL", "postgresql://forensic:forensic@localhost:5432/forensic")
    vector_backend: str = os.getenv("VECTOR_BACKEND", "local")  # local | milvus
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    # Milvus 컬렉션명. 실험·반복별 분리를 위해 런타임에 바꿀 수 있음.
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "forensic_entities")

    ner_types: list[str] = field(
        default_factory=lambda: _split(os.getenv("NER_TYPES", "인물,날짜,금액,계좌,장소"))
    )
    parse_quality_threshold: float = float(os.getenv("PARSE_QUALITY_THRESHOLD", "0.6"))
    search_max_iters: int = int(os.getenv("SEARCH_MAX_ITERS", "4"))
    topk: int = int(os.getenv("TOPK", "5"))


CFG = Config()
