# Scores

This file is just here to keep track of which method improves or worsens the score. 

To assess the score you can either do a submission or use the ```eval.py``` . This script is reverse-engineered version, and seem to always be around the right value (slightly above in fact). 

---

## Baselines

- With the base model and base dataset :  0.89559 (kaggle) 0.90___ (script)

- Nearest neighbors : 0.17229 (kaggle)

---

## Encoding experiments

- Tried using deepwalk to encode the graphs instead of spectral embedding
- Tried using bert to encode the text (didn't use the cls token but a summing aggregation, my mistake I will maybe try using the CLS token)
- Tried bert and concatenation with the original 7 features.

None of these significantly improve anything.

---

## With a bigger dataset

| method                                                       | kaggle score | ```eval.py``` score |
| ------------------------------------------------------------ | ------------ | ------------------- |
| Base                                                         | 0.84613      | 0.86___             |
| without the the feature-projecting MLP (in the denoiser)     | 0.84970      | 0.86132             |
| without the feature projecting MLP (in the denoiser) and enforcing prediction of the features in the latent space (ask Marceau for details) | 0.84292      |                     |
| without the feature projecting MLP (in the denoiser) and using conditioning in the VAE | 0.84717      | 0.8553              |

|

