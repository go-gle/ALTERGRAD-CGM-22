#!/bin/bash

# Run the command 50 times in a loop
for i in {1..250}
do
  echo "Running iteration $i..."
  python3 main_condVAE.py --hidden-dim-encoder 32 --hidden-dim-decoder 768 --latent-dim 64 --n-layers-encoder 4 --n-layers-decoder 4 --cond-hid-dim 48 --resume-training-vae --epochs-autoencoder 10 --lr 1e-3 --use-denoiser
done
