mode=$1
model="gpt-4.1-2025-04-14"
output_model_name="${model}"
export result_dir="./results/non_live_results/${output_model_name}_env_dynamics"
non_live="multi_turn"

# ===== exploration prompt =====
export USE_EXPLORATION_PROMPT="False"
# ==== end of exploration prompt =====

# ===== environment dynamics prompt =====
export USE_ENV_DYNAMICS_PROMPT="True"
export ENV_DYNAMICS_PATH="<YOUR ENV DYNAMICS PATH>"
# ==== end of environment dynamics prompt =====

if [ "$mode" == "prompt" ]; then
    # prompt mode
    bfcl generate \
        --model $model \
        --test-category $non_live \
        --num-threads 10 \
        --result-dir $result_dir
elif [ "$mode" == "function" ]; then
    # function calling
    bfcl generate \
        --model $model \
        --test-category $non_live \
        --num-threads 10 \
        --result-dir $result_dir
else
    echo "Invalid mode: $mode"
    exit 1
fi