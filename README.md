This repo contains code necessary to test LLMs for 3D manifold block design by reasoning on masked structured text representations.

- preprocess.ipynb turns the raw XML files stored in /manifolds folder into flattened text representations
- inferenceCoT.py is the inference code for the coordinate masking task
- inferencestep.py is the inference code for the step length masking task
- inferencewholetemp.py is the inference code for the whole masking task.
- finetune.py finetunes the Qwen model on the input provided by preprocess.ipynb
