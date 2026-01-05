port=<your vllm port>

model="Qwen/Qwen2.5-14B-Instruct"
export tensor_parallel_size_my=1
export VLLM_PORT=$port
export VLLM_CONFIGURE_LOGGING=0 # disable vllm logging
export PA_LR=0.1
export PA_STEPS=1
export TASK="bfcl"
output_model_name="Qwen2.5-14B-Instruct"
export result_dir="./results/non_live_results/${output_model_name}_pa_${PA_STEPS}_${PA_LR}_rebuttal_results"
non_live="multi_turn_base"

# ===== exploration prompt =====
# use this exploration to collect trajectories for dynamics extraction
export USE_EXPLORATION_PROMPT="False"
# ==== end of exploration prompt =====

# ===== environment dynamics prompt =====
export USE_ENV_DYNAMICS_PROMPT="True"
export ENV_DYNAMICS_PATH="<YOUR ENV DYNAMICS PATH>"
# ==== end of environment dynamics prompt =====

echo "Running BFCL with model: $model, test category: $non_live, result directory: $result_dir"

bfcl generate \
    --model $model \
    --test-category $non_live \
    --backend vllm \
    --num-gpus 1 \
    --num-threads 10 \
    --gpu-memory-utilization 0.9 \
    --result-dir $result_dir