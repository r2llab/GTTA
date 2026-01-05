## Grounded Test-Time Adaptation
This is the codebase to reproduce the results of the paper [Grounded Test-Time Adaptation for LLM Agents](https://arxiv.org/abs/2511.04847).

| Parametric Adaptation Framework | Non-Parametric Adaptation Framework |
|:-------------------------------:|:-----------------------------------:|
| <img src="assets/pa.png" width="400"/> | <img src="assets/npa.png" width="400"/> |

### WebArena
We adopt NNetnav's codebase for web navigation exploration and task evaluation. To reproduce our results on WebArena, please refer to [this](WebArena/README.md).

### BFCLv3
For BFCLv3 experiment, we modify our method based on the official gorilla codebase. To reproduce our results on BFCLv3, please refer to [this](BFCLv3/README.md).

### Tau-Bench
For Tau-Bench experiment, please refer to [official codebase](https://github.com/sierra-research/tau-bench) with parametric adaptation enabled.

## Citation
If you find this work useful, please cite:
```bibtex
@article{chen2025grounded,
  title={Grounded Test-Time Adaptation for LLM Agents},
  author={Chen, Arthur and Liu, Zuxin and Zhang, Jianguo and Prabhakar, Akshara and Liu, Zhiwei and Heinecke, Shelby and Savarese, Silvio and Zhong, Victor and Xiong, Caiming},
  journal={arXiv preprint arXiv:2511.04847},
  year={2025}
}
```

## License
This work is licensed under the [MIT License](LICENSE).
