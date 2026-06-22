"""오프라인용 결정론적 가짜 LLM (스모크/평가의 --fake 모드 공용).

실제 LLM 호출부만 대체하고, 에이전트 로직(라우팅·검색정책·SQL·검증)은 실제 코드로 검증.
"""
from __future__ import annotations
import hashlib
import re

DIM = 32
PERSONS = ["홍길동", "김철수", "이영희", "박민수", "정대만"]
PLACES = ["강남역", "부산", "해운대", "서울", "영등포", "여의도"]


def fake_embed(texts, kind="query"):
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        out.append([((h[i % len(h)] / 255.0) - 0.5) for i in range(DIM)])
    return out


def fake_ner(text):
    ents, seen = [], set()
    def add(val, typ):
        k = (val, typ)
        if k not in seen:
            seen.add(k); ents.append({"entity_text": val, "entity_type": typ, "context": text[:60]})
    for d in re.findall(r"\d{4}-\d{2}-\d{2}", text):
        add(d, "날짜")
    for a in re.findall(r"[\d,]+원", text):
        add(a, "금액")
    for ac in re.findall(r"\d{2,4}-\d{2,4}-\d{3,6}", text):
        add(ac, "계좌")
    for p in PERSONS:
        if p in text:
            add(p, "인물")
    for pl in PLACES:
        if pl in text:
            add(pl, "장소")
    return ents


def fake_classify(query):
    if "관련" in query:
        return {"type": "fuzzy", "reason": "fake"}
    if any(k in query for k in ["가장", "몇", "순", "카운트", "큰"]):
        return {"type": "aggregate", "reason": "fake"}
    if "문서" in query:
        return {"type": "relational", "reason": "fake"}
    return {"type": "lookup", "reason": "fake"}


def _token(query):
    for p in PERSONS + PLACES:
        if p in query:
            return p
    m = re.search(r"[가-힣]{2,4}", query)
    return m.group(0) if m else ""


def fake_sql(query, hints=None, relax=False):
    base = ("SELECT e.entity_text, e.entity_type, e.doc_id, d.file_name "
            "FROM entities e JOIN documents d ON e.doc_id = d.doc_id")
    if hints:
        vals = ",".join("'" + h.replace("'", "") + "'" for h in hints)
        return f"{base} WHERE e.entity_text IN ({vals})"
    tok = _token(query)
    if not tok:
        return base
    if relax:
        return f"{base} WHERE e.entity_text LIKE '%{tok}%' OR e.context LIKE '%{tok}%'"
    return f"{base} WHERE e.entity_text = '{tok}'"


def fake_parse_file(path):
    """오프라인 평가용 비텍스트 파일 파서. 외부 OCR/STT API 호출을 막는다."""
    import os
    name = os.path.basename(path)
    return f"오프라인 파서 결과: {name}"


def apply_fakes():
    """에이전트 모듈의 LLM 바인딩을 가짜로 교체."""
    import agent1.graph as a1g
    import agent1.tools as a1t
    import agent2.tools as a2t
    a1g.embed = fake_embed
    a1t.run_ner = fake_ner
    a1t.upstage_document_parse = fake_parse_file
    a1t.vlm_ocr_pdf = fake_parse_file
    a1t.vlm_ocr_image = fake_parse_file
    a1t.clova_speech = fake_parse_file
    a1t.whisper = fake_parse_file
    a1t.vlm_caption_image = lambda path: "테스트 이미지 캡션"
    a1t.PARSERS.update({
        "upstage_document_parse": fake_parse_file,
        "vlm_ocr_pdf": fake_parse_file,
        "vlm_ocr_image": fake_parse_file,
        "clova_speech": fake_parse_file,
        "whisper": fake_parse_file,
    })
    a2t.embed = fake_embed
    a2t.classify_query = fake_classify
    a2t.text_to_sql = fake_sql
