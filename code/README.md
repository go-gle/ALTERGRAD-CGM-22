# ALTEGRAD 2024 - CGM team code

## Organization of the directory.

This repo holds the code associated to teh ALTEGRAD 2024 Data Challenge. It is organized as so :
```
.
├── experiments
│   ├── CLIP
│   ├── Cond_CLIP
│   ├── condVAE
│   ├── finetune_llm_llama
│   ├── finetune_llm_phi3_unsuccessful
│   └── MLP
└── src
    └── baseline

```

The `src` directory holds utilitaries common to several experiments, including some of the baseline `.py` files we were handed, some of which we did slightly modify. The `experiments` directory holds the various architectures or tricks we attempted to solve the problem. 

## Reproducing experiments

In an environment such as `altegrad-lab5` as established during the labs, just copy the `data/` folder in the associated experiment subdir (e.g. `CLIP/`) and simply run the associated `main___.py` you wish to check.

