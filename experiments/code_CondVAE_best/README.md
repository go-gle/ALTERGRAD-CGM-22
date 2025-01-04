# README of how to run the files in this dir.

This was built on top of Gleb's `cond_autoencoder.py` (which I didn't touch but for minor changes to help parse arguments). This folder is self contained but for the data. You should use the original data, or you can play around with generated data. At first sight, a lot of generated data doesn't seem to improve things (it does stabilize the results, but so far our best run was a lucky one, and a lot of data probably doesn't allow for that). I you want to play around you can.

Table of contents:

- Notes on the commands

- Quick overview of some of the nomenclature and file organization.

- How to use a pretrained model.

- How to train

  - CondVAE

  - Denoiser

    

---

## What's what

- The `temp/` directory just holds temporary checkpoints of the models (both the CondVAE and the denoiser depending on what's being trained) appending the names with a number that is random (drawn before any seed is set) in order not to mix up the files. **This means you can run several trainings at once without worrying**. One training of either the VAE or the denoiser takes around 500 MB of vRAM (fyi).

- The `runs/` dir holds all the files generated when training. It looks like this:

  - ```
    runs
    ├── args
    │   └── args_0-21641.txt
    ├── autoencoders
    │   └── autoencoder_condVAE_0-21641.pth.tar
    ├── denoisers
    └── output_csvs
        └── condVAE_0-21641.csv
    ```

    I left the files corresponding to our best model so far in there. 

  - The suffix `*file*_0-21641.*ext*` means that the file was part of the model that lead to an evaluation of 0.21641 using the `eval_final.py` util file that is used at the end of inference in `main_condVAE.py`. It will also be displayed in the terminal when running the main file. **This isn't the kaggle score, but it correlates to it;** For instance 0.21641 gave 0.1231,  0.395 gave 0.305 and so on. It's not necessarily .09 above, but it usually is.

  - `args/` hold txt files with the corresponding parameters of each model. If you want to run a pretrained model, see next section and use these files.

- `condVAE_attempts.md` reports the scores and the architectures corresponding to the trainings of the CondVAE

- `denoiser_attempts.md` is the same for the denoiser. TLDR : i can't make the denoiser improve shit (I still need to play around with the timesteps though)

- Some files are somewhat useless but still I am leaving them here in just in case. It probably breaks the script. Such files are `bert_utils.py` and `deepwalk_utils.py`.

---

## Notes on some custom arguments:

In the following : 

```sh
python3 main_condVAE.py --use-denoiser --train-autoencoder --use-cond-denoising --train-denoiser
```

All of these do exactly the opposite (they are `store_false` arguments). I stuck with this convention because this is how the base arguments were. Just beware of that, or just read the code when in doubt.

---

## Using a pretrained model

Either to do inference or to train a denoiser once a CondVAE has been trained (recommended route).

**example 1 ** : train a denoiser using the best autoencoder so far* :

- identify the files needed : `args_0-21641.txt`, `autoencoder_condVAE_0-21641.pth.tar`.

- copy `autoencoder_condVAE_0-21641.pth.tar` into the same dir as `main_condVAE.py` and **rename** it `autoencoder_condVAE.pth.tar`.

- The mandatory arguments are the ones relevant to the autoencoder present in  `args_0-21641.txt`. They are:

  - ```sh
    python3 main_condVAE.py --hidden-dim-encoder 32 --hidden-dim-decoder 768 --latent-dim 64 --n-layers-encoder 4 --n-layers-decoder 4 --cond-hid-dim 48 --train-autoencoder 
    ```

    - `train-autoencoder` freezes the autoencoder and loads it from the main dir. Hence the renaming above.

- If you want to pass in custom args for the denoiser, please do so like you would normally on top of the previous command :

  - ```sh
    python3 main_condVAE.py --hidden-dim-encoder 32 --hidden-dim-decoder 768 --latent-dim 64 --n-layers-encoder 4 --n-layers-decoder 4 --cond-hid-dim 48 --train-autoencoder --epochs-denoise 50000 --hidden-dim-denoise 512 --dim-condition 32 --n-layers_denoise 3 --timesteps 500 --lr 1e-3
    ```

**example 2** *To simply infer with both a denoiser and an autoencoder*

- take the files from `autoencoders/` and `denoisers/`, copy them in the main dir and rename them **exactly** `autoencoder_condVAE.pth.tar` and `denoise_model.pth.tar`

- run the main.

Or you can use the `eval.py` but i don't want to make a doc for it, just read the argparse in there if you insist.

---

## Training

**Outline** : Train the CondVAE on its own first, as it is the "clutch" one. The denoiser at best adds uncertainty as to the quality of the predictions, and at worse degrades them. The argument `--use-denoiser` will make the main ignore all the parts related to the denoiser.

### Training the condVAR

- To train a CondVAE, here is all you have to run. Just replace the arguments of interest by the ones you want to try:

```sh
python3 main_condVAE.py --hidden-dim-encoder 32 --hidden-dim-decoder 768 --latent-dim 64 --n-layers-encoder 4 --n-layers-decoder 4 --cond-hid-dim 48 --use-denoiser --lr 1e-3 --epochs-autoencoder 50000
```

Note that `--epochs-autoencoder 50000` is a dumb number simply here to trigger the early stopping procedure (by default at 30, editable with `--				early-stopping-rounds`).

- To train it on a dataset that is not the base one, simply add the argument `--dataset` along the name of the corresponding dir within the main dir : 

```sh
python3 main_condVAE.py --hidden-dim-encoder 32 --hidden-dim-decoder 768 --latent-dim 64 --n-layers-encoder 4 --n-layers-decoder 4 --cond-hid-dim 48 --use-denoiser --lr 1e-3 --epochs-autoencoder 50000 --dataset data_gen6
```



This will write the new `autoencoder_0-score.pth.tar`,  output csv `condVAE_0-score.csv` and corresponding arguments `args_0-score.txt` in the relevant subdirs of `runs/`.

### Training the denoiser

See the section ***Using a pretrained model*** above.



For any further questions or if anything doesn't run, ask me on whatsapp.
