# README of how to run the files in this dir.

Very similar to the README.md in experiments/cond_VAE_best. Ideally, read it before



---

## What's what

- The `temp/` directory just holds temporary checkpoints of the models (depending on what's being trained) appending the names with a number that is random (drawn before any seed is set) in order not to mix up the files. **This means you can run several trainings at once without worrying**.

- The `runs/` dir holds all the files generated when training. It looks like this:

  - ```
    runs
    ├── args
    │   └── args_0-XXXXX.txt
    ├── mlps
    │   └── MLP_0-XXXXX.pth.tar
    ├── denoisers
    └── output_csvs
        └── MLP_0-XXXXX.csv
    ```

    I left the files corresponding to our best model so far in there. 

  - The suffix `*file*_0-XXXX.*ext*` means that the file was part of the model that lead to an evaluation of 0.21641 using the `eval_final.py` util file that is used at the end of inference in `main_condVAE.py`. It will also be displayed in the terminal when running the main file. **This isn't the kaggle score, but it correlates to it;** For instance 0.21641 gave 0.1231,  0.395 gave 0.305 and so on. It's not necessarily .09 above, but it usually is.

  - `args/` hold txt files with the corresponding parameters of each model. If you want to run a pretrained model, see next section and use these files.

- `mlp_attemps.md` reports the scores and the architectures corresponding to the trainings of the MLPs

  

- Some files are somewhat useless but still I am leaving them here in just in case. It probably breaks the script. Such files are `bert_utils.py` and `deepwalk_utils.py`.

---

In this folder you will find 3 `main_` training scripts corresponding to:

- training of a simple MLP (ok tier)
- training of an MLP with a skip connection between the output of the first layer and the input of the 2nd to last layer. This did not improve the results nor deteriorated them.
- training of a "mixture" of 2 MLPs. According to a threshold `--cutoff`, graphs with a number of nodes above or below will go into 2 separately trained MLPs. This gave the best kaggle score of .11602 despite not being the best `eval` score (though close to it).

---

## Train a simple MLP

In the following : 

```sh
python3 main_MLP.py --lr 1e-3 --epochs-mlp 50000 --architecture 16,64,256,1024,2048 --batch-size 256
```

All of these do exactly the opposite (they are `store_false` arguments). I stuck with this convention because this is how the base arguments were. Just beware of that, or just read the code when in doubt.

---

## Train a mix of 2 MLPs with the same architectures but that focus on 2 different sizes of graphs

There is a cutoff that depends on the number of nodes, to determine which mlp is the feature vector going to get treated by. This has led to our best kaggle and eval.py score. **Mind that the cutoff is in a standardized form**, e.g. 0 (the best cutoff so far, see `mlp_attempts.md`) will make the effective cutoff be 30 nodes.

```sh
python3 main_mix_MLP.py --lr 1e-3 --epochs-mlp 50000 --architecture 16,64,256,1024,2048 --batch-size 256 --cutoff 0
```

