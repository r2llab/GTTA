BASE_URL="<YOUR BASE URL>"
export WA_SHOPPING="$BASE_URL:8082"
export WA_SHOPPING_ADMIN="$BASE_URL:8083/admin"
export WA_REDDIT="$BASE_URL:8080"
export WA_GITLAB="$BASE_URL:9001"
export WA_WIKIPEDIA="$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:443"
export WA_HOMEPAGE="$BASE_URL:80"
export TOKENIZERS_PARALLELISM=false
export BROWSERGYM_EXPERIMENT_TIMEOUT=600 # 10 minutes

# == using openai ==
model_name="<YOUR MODEL NAME>"
tag="<YOUR TAG>"
export USE_ENV_DYNAMICS=true # this is for step 3 of non-parametric test-time adaptation
export ENV_DYNAMICS_PATH="<YOUR ENV DYNAMICS PATH>" # please specify the processed environment dynamics path

python src/run_agent.py \
  --use_openai \
  --provider openai \
  --model $model_name \
  --temperature 0.0 \
  --result_dir webarena_results/${model_name}_${tag}_temp=0.0 \
  --instruction_path src/agent/prompts/jsons/p_cot_llama_action_history_bgym_env_dynamics.json \
  --agent_name webarena_${model_name}_${tag}_temp=0.0 \
  --n_jobs 15 \
  --data webarena