## BFCLv3

### Environment Setup
Follow the environment setup [guide](berkeley-function-call-leaderboard/README.md) for BFCLv3.

### Parametric Test-Time Adaptation

**Setup:**
1. Replace the `model_runner.py` file in vllm's worker directory with our modified version:
```bash
cp vllm/model_runner.py "$(python -c 'import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), "worker"))' | tail -n 1)"
```

### Non-Parametric Test-Time Adaptation

#### Step 1: Exploration Goals Synthesis
Use the [jupyter notebook](berkeley-function-call-leaderboard/notebooks/synthesize_exploration.ipynb) to synthesize the exploration goals.

#### Step 2: Exploration Goals Execution
Put the exploration goals in to `data/` folder and run `bfcl generate` command (see [example script](berkeley-function-call-leaderboard/example_scripts/run_bfcl_api.sh)) with `USE_EXPLORATION_PROMPT=True` flag.

#### Step 3: Environment Dynamics Extraction and Filtering
Use the [jupyter notebook](berkeley-function-call-leaderboard/notebooks/extract_dynamics.ipynb) to extract and filter the environment dynamics.

#### Step 4: BFCLv3 Task Evaluation
Execute tasks with the environment dynamics by turning `USE_ENV_DYNAMICS_PROMPT=True` flag and setting `ENV_DYNAMICS_PATH` to the path of the environment dynamics. An example script is provided in [run_bfcl_api.sh](berkeley-function-call-leaderboard/example_scripts/run_bfcl_api.sh).

#### Step 5: Evaluation
Use the official BFCLv3 evaluation script to evaluate the results. An example is as follows:
```bash
bfcl evaluate --model MODEL_NAME --result-dir RESULT_DIR
```
