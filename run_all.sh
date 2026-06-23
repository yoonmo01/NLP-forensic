#!/usr/bin/env bash
# B -> C -> D 순차 실험 자동화. Run A(현재 진행 중)가 끝나면 자동으로 시작한다.
# 한 번에 하나씩만 실행(메모리 보호). 중간 실패해도 다음 단계는 계속 진행.
# 사용: nohup bash run_all.sh > run_all.log 2>&1 &
cd "$(dirname "$0")"
source .venv/bin/activate

# 0) Run A 끝날 때까지 대기 (run.log에 DONE 이 찍히면 진행)
until grep -q "DONE" run.log 2>/dev/null; do sleep 60; done
echo "[auto] A done -> B"

# B) qwen graph : toolcall vs graph 라우팅 비교 (Agent1 효과 분리)
python run_experiment.py --repeats 1 --mode graph > run_B_graph.log 2>&1
echo "[auto] B done"

# C) gemma4 toolcall : 백본 비교
sed -i 's/^LLM_MODEL=.*/LLM_MODEL=gemma4:26b/' .env
sed -i 's/^VLM_MODEL=.*/VLM_MODEL=gemma4:26b/' .env
python run_experiment.py --repeats 1 --mode toolcall > run_C_gemma.log 2>&1
# .env 를 qwen 으로 복구
sed -i 's/^LLM_MODEL=.*/LLM_MODEL=qwen3.5:35b-a3b/' .env
sed -i 's/^VLM_MODEL=.*/VLM_MODEL=qwen3.5:35b-a3b/' .env
echo "[auto] C done"

# D) qwen 일관성 : 적재는 결정론(DB 고정), 질의만 temp 0.7 로 3회 -> EQ3 응답 변동 측정
QUERY_TEMPERATURE=0.7 python run_experiment.py --repeats 3 --mode toolcall > run_D_temp07.log 2>&1
echo "[auto] D done"

echo "[auto] ALL DONE (B/C/D)"
