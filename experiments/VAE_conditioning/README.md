These files are the ones you need to insert in a copy of the original "code" repo in order to play around with the conditioning I tried.

Goal : Improve the latent space representation

Method : Condition the VAE, using concatenation and projection of the features.

# How it was done

Because the original encoder involves convolutions (GIN), it doesn't make sense to concatenate everything in this part. To condition, I chose to encode the features separately in their own MLP. Then, the output of both the GIN and the features MLP are projected together throug another MLP. The output of that last MLP is the latent space.

Relevant details :

- the features were standard-scaled (makes sense for NNs, and I am not doing it just for this experiment. Also it yields equal performance with the baseline when not changing anything else so we can go ahead).
- The latent space dimension was unchanged to assess if we improved the results.
- I tried both an upscaling of the features in their first MLP (going from 7 to 16) as well as just projecting them (from 7 to 8) and reducing them (7 to 4). 

# Results

In each case, this yielded results comparable to our "usual" results, i.e. in the ball park of ~0.84 on kaggle, using a synthetised dataset as per usual (6:1 ratio of synthetic:real data).

