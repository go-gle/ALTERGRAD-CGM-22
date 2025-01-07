This is a simple outline for the report. Just trying to draft some of the parts and their content. Feel free to move things around, delete, replace or add stuff as you wish.

---

# 1. Introduction

general paper style intro to the problem :

- the general problem with the reference of the original paper
- the simplified version we were handed (fixed 7 features, small dataset)

# 2. Related work

Honestly don't really know what to put in there. Maybe:

- briefly talk about the original paper and situate why they use an VAE and a diffusion model over say a normalizing flows model.
- CLIP

# 3 Approach & results

We probably should deviate from the typical pattern of having one part for the approach and one part for the results, as our approach was determined by the results we had in the process.

## 3.1 Data modelization

- tried a deepwalk embedding instead of a spectral one
- tried encoding the sentences as a whole with BERT
- Finally stuck with just spectral + the 7 extracted features
- Generated extra graphs (descibing the procedure). Improved the original model but not the following tweaks (CondVAE, MLPs)

## 3.2 Metric 

- Because their is a leaderboard we tried and somewhat succeeded in reverse engineering the evaluation metric.

## 3.3 Original model tuning

- marginal improvements when using some generated graphs
- incorporation of early stopping procedure
- couldn't improve the metric much more.

## 3.4 Training a LoRA to directly spit the edgelist

- Marceau : Phi3-4k-mini-instruct : unsuccessful training
- Channdeth : successful training, probelm with the context (the bigger graphs have edgelists of up to 7k tokens, making small LLMs unsuable)

## 3.5 Incorporating conditioning in the VAE

- First noticeable improvements

## 3.6 Incorporating contrastive learning objective

- sequential training in the VAE
- joint training of the whole VAE
- Usage of the yielded feature encoder
- incorporation with the conditioned VAE
- unsignificant results at best, worse ones at worst

## 3.7 Tuning of the CondVAE and ablations

- Found that the denoiser made things worse than pure randomness. Add part on the tuning of the denoiser.
- KNN instead was marginally better
- found that a small latent space gave better results, suggesting that the conditioning was more important, suggesting a simple MLP would yield as good results if not better ones.

## 3.8 Use of MLPs and metric abuse

- Simple MLP was as good as the conditionnal VAE and sometimes marginally better
- Tried MLP with skip connections, unsignificant improvements
- Tries to split the work between 2 different MLPs, small but noticeable improvements
- noticed that the reconstruction loss prevented the model from learning further from the data, leading us to use very large early stopping patience and eventually getting rid of the validation procedure to also merge training and validation set, leading us to our best result so far

# 4. Lessons

- The diffusion model probably suffered from:
  - the small size of the original dataset both in its training and in the latent space it aimed to recover
  - the latent space of the VAE itself that did not recover clutch information
- Because we had a simplified task we could abuse the metrics to produce a simple yet performant model
