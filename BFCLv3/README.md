## BFCLv3

### Environment Setup
Follow the environment setup [guide](BFCLv3/berkeley-function-call-leaderboard/README.md) for BFCLv3.

### Parametric Test-Time Adaptation

**Prerequisites:**
- Install `vllm==0.7.2`

**Setup:**
1. Replace the `model_runner.py` file in vllm's worker directory with our modified version:
```bash
cp vllm/model_runner.py "$(python -c 'import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), "worker"))' | tail -n 1)"
```
Execute the following command to start 

### Non-Parametric Test-Time Adaptation

#### Step 1: Exploration Goals Synthesis
Use the [jupyter notebook](BFCLv3/berkeley-function-call-leaderboard/notebooks/synthesize_exploration.ipynb) to synthesize the exploration goals.

#### Step 2: Environment Dynamics Extraction and Filtering
Use the [jupyter notebook](BFCLv3/berkeley-function-call-leaderboard/notebooks/extract_dynamics.ipynb) to extract and filter the environment dynamics.

#### Step 3: BFCLv3 Task Evaluation
Use the official BFCLv3 evaluation script to evaluate the results. An example script is provided in [run_bfcl_api.sh](BFCLv3/berkeley-function-call-leaderboard/run_bfcl_api.sh).
An example running script is provided in [here](BFCLv3/berkeley-function-call-leaderboard/run_bfcl_api.sh).
