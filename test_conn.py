"""게이트웨이 연결·모델·임베딩 점검 (키 비노출)."""
from config import CFG
from llm import client, embed

print("== 설정 ==")
print("LLM_MODEL     =", CFG.llm_model)
print("BASE_URL      =", CFG.base_url)
print("KEY_SET       =", bool(CFG.api_key), "(len=%d)" % len(CFG.api_key))
print("EMBED_BACKEND =", CFG.embed_backend)
print("EMBED_BASE_URL=", CFG.embed_base_url)
print("EMBED_KEY_SET =", bool(CFG.embed_api_key))

print("\n== 모델 목록 ==")
try:
    ids = [m.id for m in client().models.list().data]
    print("count=", len(ids))
    print(ids[:40])
    print("gpt-4.1-mini 있음?", any("gpt-4.1-mini" in i for i in ids))
except Exception as e:
    print("models.list 오류:", repr(e))

print("\n== 채팅 1회 ==")
try:
    r = client().chat.completions.create(
        model=CFG.llm_model, temperature=0,
        messages=[{"role": "user", "content": "한 단어로만 답해: 대한민국 수도?"}])
    print("OK:", (r.choices[0].message.content or "").strip()[:60])
except Exception as e:
    print("채팅 오류:", repr(e))

print("\n== 임베딩 1회 ==")
try:
    v = embed(["연결 테스트"])
    print("OK backend=%s dim=%d" % (CFG.embed_backend, len(v[0])))
except Exception as e:
    print("임베딩 오류:", repr(e))
