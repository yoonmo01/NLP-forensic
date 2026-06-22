# Forensic 2-Agent System (프로토타입)

도구 호출·하이브리드 검색·Text-to-SQL·HITL을 결합한 디지털 포렌식 수사 질의 2-에이전트 시스템.
논문(기말 Full Draft) Methodology/Experiments의 실행 구현체.

## 구성

```
forensic_agents/
  config.py            설정(.env 로드)
  llm.py               OpenAI 호환 LLM 래퍼(chat/json/embed/vision)
  storage/
    relational.py      SQLite (documents, entities) — 논문 PostgreSQL 스키마 호환
    vectorstore.py     하이브리드 검색: LocalHybridStore(BM25+밀집, 기본) / MilvusStore(옵션)
  agent1/              수집·구조화 (라우팅·실패복구 LangGraph)
    tools.py           detect/parse(pdf,hwp,txt,vlm,whisper)/품질평가/NER/지문
    graph.py           detect→parse→assess→(fallback)→ner→persist
  agent2/              수사 질의 (ReAct + 검색정책 LangGraph)
    tools.py           분류/하이브리드 해소/Text-to-SQL/실행/검증/포맷
    graph.py           classify→(fuzzy?)resolve→gen_sql→exec→decide(narrow↔broaden)→verify→format
  run_ingest.py        Agent 1 실행 (디렉터리 적재)
  run_query.py         Agent 2 실행 (HITL 대화 루프)
  run_eval.py          평가 하네스 (분류정확도/Recall@k/MRR/SQL/Answer)
  test_smoke.py        오프라인 스모크 테스트
  mock_llm.py          오프라인 가짜 LLM
  data/sample/         합성 한국어 샘플(개발용)
  eval/queries.jsonl   표준 질의셋(정답 라벨)
```

## 설치

```powershell
python -m pip install -r requirements.txt
copy .env.example .env   # .env 에 OPENAI_API_KEY 등 입력
```

`.env` 핵심:
- `OPENAI_API_KEY` (필수), `OPENAI_BASE_URL`(HAI-GPT 등 호환 엔드포인트 시), `LLM_MODEL=gpt-4.1-mini`
- `VECTOR_BACKEND=local`(기본) 또는 `milvus`

## 실행 (실제 LLM)

```powershell
# 1) 수집: 압수 디렉터리 → DB+벡터색인
python run_ingest.py --input-dir data/sample
#    실제 데이터: --input-dir "D:\\seized\\user1_Cdrive"

# 2) 질의: HITL 루프
python run_query.py

# 3) 평가: 논문 수치 산출
python run_eval.py
```

## 실행 (오프라인 검증, API 불필요)

```powershell
python test_smoke.py        # 파이프라인 배선 검증
python run_eval.py --fake   # 로직 검증(가짜 임베딩이라 검색 수치는 무의미)
```

## Milvus 백엔드(옵션, 논문 최종 스택)

Milvus Lite는 Windows 미지원. Docker 또는 WSL 사용:
```powershell
# Docker Desktop 필요
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest  # 또는 milvus standalone compose
```
`.env`에서 `VECTOR_BACKEND=milvus`, `MILVUS_URI=http://localhost:19530` 설정 후 동일하게 실행.
(WSL Ubuntu에서는 `pip install pymilvus` 후 Milvus Lite 사용 가능.)

## 논문 ↔ 구현 매핑

| 논문(Full Draft) | 구현 |
|---|---|
| 3.2 라우팅·실패복구 수집 에이전트 | `agent1/graph.py` (conditional edge `route_after_assess` = 의사결정 지점, fallback) |
| 3.3 ReAct 질의 + 검색정책 | `agent2/graph.py` (`n_decide` narrow↔broaden) |
| 3.4 하이브리드 색인 | `storage/vectorstore.py` (BM25+밀집 RRF) |
| Table 2/4 도구 설계 | `agent1/tools.py`, `agent2/tools.py` |
| 4.4 평가 지표 | `run_eval.py` |

## 한계(프로토타입)

- 관계형은 SQLite(논문 PostgreSQL 호환), 기본 검색은 로컬 하이브리드(논문 Milvus 호환).
- HWP는 olefile 간이 추출/LibreOffice fallback, 스캔 PDF는 pypdfium2+VLM(옵션 설치 시).
- MP4 STT는 선택 확장(Pending Work Plan).
