"""실험 전 사전 점검 — 5개 유형을 실제 파일 1개씩 파서에 직접 통과시켜
어떤 구성요소(VLM/STT/PDF/HWP/임베딩)가 깨지는지 즉시 확인한다.
LLM 에이전트(toolcall) 없이 파서 함수를 직접 호출하므로 빠르고 원인 격리가 쉽다.

전제: data/eval_set/ 와 .env(sqlite/local + Ollama)가 준비돼 있어야 함.
사용: python preflight.py
"""
from __future__ import annotations
import glob
import os
import sys
import traceback

OK, FAIL = "[ OK ]", "[FAIL]"
problems: list[str] = []


def step(t): print(f"\n=== {t} ===", flush=True)


def first_of(ftype: str):
    import agent1.tools as T
    for p in sorted(glob.glob("data/eval_set/*")):
        if os.path.isfile(p) and T.detect_file_type(p) == ftype:
            return p
    return None


def try_parse(path: str):
    """STRATEGIES 순서대로 시도(실제 ingest와 동일한 fallback). (전략, 품질, 텍스트) 반환."""
    import agent1.tools as T
    ft = T.detect_file_type(path)
    strats = T.STRATEGIES.get(ft, ["txt"])
    last = None
    for s in strats:
        try:
            txt = T.PARSERS[s](path)
            q = T.assess_extraction_quality(txt)
            return s, q, txt
        except Exception as e:
            last = f"{s} 실패: {type(e).__name__}: {e}"
    raise RuntimeError(last or "사용 가능한 전략 없음")


def main() -> int:
    # 0) 백엔드
    step("0) 백엔드 (.env)")
    from config import CFG
    print(f"  relational={CFG.relational_backend} (기대 sqlite) / vector={CFG.vector_backend} (기대 local)")
    print(f"  llm={CFG.llm_model} @ {CFG.base_url}")
    print(f"  vlm={CFG.vlm_model or '(메인 LLM)'} / stt={CFG.stt_backend}/{CFG.whisper_model} dev={CFG.stt_device}")
    if CFG.relational_backend == "postgres" or CFG.vector_backend == "milvus":
        problems.append("백엔드가 postgres/milvus — .env에서 sqlite/local 확인")

    # 1) 유형별 실제 파싱 (txt → pdf → hwp → image[VLM] → audio[STT])
    for ft, note in [("txt", "인코딩"), ("pdf", "pypdf→실패시 VLM OCR"),
                     ("hwp", "libreoffice→olefile"), ("image", "VLM OCR/캡션"),
                     ("audio", "faster-whisper STT")]:
        step(f"파싱: {ft}  ({note})")
        p = first_of(ft)
        if not p:
            print(f"  (data/eval_set에 {ft} 샘플 없음 — 건너뜀)")
            continue
        print(f"  file: {os.path.basename(p)}")
        try:
            s, q, txt = try_parse(p)
            snip = " ".join((txt or "").split())[:160]
            verdict = OK if (txt and txt.strip()) else f"{OK}(빈 텍스트→캡션 단계로)"
            print(f"  {verdict} 전략={s} 품질={q} chars={len(txt or '')}")
            if snip:
                print(f"        «{snip}»")
        except Exception as e:
            print(f"  {FAIL} {e}")
            problems.append(f"{ft} 파싱 실패 — {e}")

    # 2) 임베딩(BGE-M3) — 벡터 검색 전제
    step("임베딩 BGE-M3 (최초 1회 다운로드)")
    try:
        from llm import embed
        v = embed(["강수민 대리 거래명세서 2,300,000원"], kind="passage")
        print(f"  {OK} dim={len(v[0])} (BGE-M3 기대 1024)")
    except Exception as e:
        print(f"  {FAIL} {type(e).__name__}: {e}")
        traceback.print_exc()
        problems.append("임베딩 실패 — sentence-transformers/BGE-M3 확인")

    # 3) LLM tool-calling (에이전트 동작 게이트)
    step("LLM tool-calling")
    try:
        from agent1.toolcall import TOOLSPECS
        from llm import client
        r = client().chat.completions.create(
            model=CFG.llm_model, temperature=0, tools=TOOLSPECS, tool_choice="auto",
            messages=[{"role": "user", "content": "detect_type 도구를 호출하라."}])
        tc = r.choices[0].message.tool_calls
        print(f"  {OK} tool_call={tc[0].function.name}" if tc else f"  {FAIL} tool_calls 없음")
        if not tc:
            problems.append("LLM tool-calling 안 됨")
    except Exception as e:
        print(f"  {FAIL} {type(e).__name__}: {e}")
        problems.append("LLM 호출 실패 — Ollama/OPENAI_BASE_URL 확인")

    # 요약
    step("요약")
    if problems:
        print(f"  {FAIL} 미해결 {len(problems)}건:")
        for x in problems:
            print(f"     - {x}")
        print("\n  → 위 항목 먼저 고친 뒤 run_experiment.py 진행")
        return 1
    print(f"  {OK} 전 유형 통과 → run_experiment.py 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
