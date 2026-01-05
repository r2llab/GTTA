BASE_URL="<YOUR BASE URL>"
export WA_SHOPPING="$BASE_URL:8082"
export WA_SHOPPING_ADMIN="$BASE_URL:8083/admin"
export WA_REDDIT="$BASE_URL:8080"
export WA_GITLAB="$BASE_URL:9001"
export WA_WIKIPEDIA="$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:443"
export WA_HOMEPAGE="$BASE_URL:80"
export TOKENIZERS_PARALLELISM=false

export ENV_DYNAMICS_EXPLORATION=true

# ======= on WebArena ======
export VLLM_PORT=8001
python src/run_exploration.py \
  --provider vllm \
  --model Qwen/Qwen2.5-14B-Instruct \
  --result_dir ./exploration_results_env_dynamics_all_num_seeds=10_webarena_qwen14b \
  --filter_dir ./exploration_filtered_env_dynamics_all_num_seeds=10_webarena_qwen14b \
  --seed_dir seed_states_webarena/ \
  --exploration_size_per_seed 10 \
  --temperature 1.0 \
  --n_jobs 15 --use_personas \
  --environment_type webarena