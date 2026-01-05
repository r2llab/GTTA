## WebArena

### Environment Setup
Follow the environment setup [guide](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/webarena/README.md) for WebArena. We adopt [NNetnav](https://github.com/MurtyShikhar/NNetnav)'s codebase for WebArena exploration.

### Parametric Test-Time Adaptation

**Prerequisites:**
- Install `vllm==0.7.2`

**Setup:**
1. Replace the `model_runner.py` file in vllm's worker directory with our modified version:
```bash
cp vllm/model_runner.py "$(python -c 'import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), "worker"))' | tail -n 1)"
```

### Non-Parametric Test-Time Adaptation

#### Step 1: Persona-based Exploration and Environment Dynamics Extraction
```bash
bash WebArena/run_exploration.sh
```

#### Step 2: Environment Dynamics Filtering
Use the [jupyter notebook](WebArena/notebooks/dynamic_modelling.ipynb) to filter the environment dynamics.

#### Step 3: WebArena Task Evaluation
```bash
bash WebArena/run_webarena.sh
```

### Evaluation
Use the [jupyter notebook](WebArena/evaluation/eval_webarena_analysis.ipynb) to evaluate the results.
