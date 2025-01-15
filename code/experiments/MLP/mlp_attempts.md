Here is the overview for the architectural tuning of the MLP. Example of running command:

```sh
python3 main_MLP.py --lr 1e-3 --epochs-mlp 50000 --architecture 16,64,512,768,1024
```

Seeds are now useful and do imply reproducibility. Let's not put them.



| architecture                                                 | lr   | dataset | batchsize | eval.py try1 | try2    | try3    | kaggle |
| ------------------------------------------------------------ | ---- | ------- | --------- | ------------ | ------- | ------- | ------ |
| 16,32,64                                                     | 1e-3 | og      | 256       | 0.24983      | 0.24324 |         |        |
| 16,64,512,768,1024                                           | 1e-3 | og      | 256       | 0.22086      | 0.22549 | 0.22199 |        |
| 16,64,256,768,1024,1024,256                                  | 1e-3 | og      | 256       | 0.23569      | 0.22583 | 0.23047 |        |
| **16,64,256,1024,2048**                                      | 1e-3 | og      | 256       | 0.22897      | 0.21794 | 0.22573 |        |
| 16,64,256,512,1024,2048                                      | 1e-3 | og      | 256       | 0.23279      | 0.23104 | 0.23296 |        |
| 2048                                                         | 1e-3 | og      | 256       | 0.23551      | 0.23289 | 0.23510 |        |
| 16,64,256,1024,4096                                          | 1e-3 | og      | 256       | 0.23071      | 0.23376 | 0.22306 |        |
| 16,32,64,128,256,512,1024,2048                               | 1e-3 | og      | 256       | 0.22918      | 0.23214 | 0.22706 |        |
| 16,32,64,128,256,512,1024,2048,4096                          | 1e-3 | og      | 256       | 0.24467      | 0.24843 | 0.24706 |        |
|                                                              |      |         |           |              |         |         |        |
| 16,64,256,1024,2048                                          | 1e-3 | og      | 32        | 0.22857      | 0.21986 | 0.21830 |        |
| 16,64,256,1024,2048                                          | 1e-3 | og      | 64        | 0.22571      | 0.23005 | 0.23639 |        |
| 16,64,256,1024,2048                                          | 1e-3 | og      | 256       | 0.22479      | 0.22243 | 0.22044 |        |
| 16,64,256,1024,2048                                          | 1e-3 | og      | 1024      | 0.24934      | 0.22162 | 0.23854 |        |
| 16,64,256,1024,2048                                          | 1e-3 | og      | 4096      | 0.25409      | 0.23178 | 0.23218 |        |
|                                                              |      |         |           |              |         |         |        |
| the noise induced by small batch sizes seem interesting      |      |         |           |              |         |         |        |
| however we do have consistently competitive results with 256 |      |         |           |              |         |         |        |

Now we had a skip connection between the 1st layer and 2nd to last layer to see if it does anything.	

| architecture        | lr   | dataset | Skip ? | eval.py try1 | try2    | try3    | kaggle |
| ------------------- | ---- | ------- | ------ | ------------ | ------- | ------- | ------ |
| 16,64,256,1024,2048 | 1e-3 | og      | yes    | 0.22533      | 0.22052 | 0.22932 |        |
| 16,64,256,1024,2048 | 1e-3 | og      | no     | 0.22798      | 0.22471 | 0.22490 |        |

Probably nothing significant.

Now let's split the work between 2 MLPs: one for small (<=30 nodes) graphs and one for big (>30 nodes) graphs. Because we standard scaled the data this means above or under 0. The mean actually 30.6, so this is perfect (30 means a value of less than 0).

| architecture        | lr   | dataset | cutoff | eval.py try1 | try2        | try3    | kaggle  |
| ------------------- | ---- | ------- | ------ | ------------ | ----------- | ------- | ------- |
| 16,64,256,1024,2048 | 1e-3 | og      | -0.5   | 0.23091      | 0.22193     | 0.23106 |         |
| 16,64,256,1024,2048 | 1e-3 | og      | 0      | 0.22824      | **0.21655** | 0.23082 | 0.11602 |
| 16,64,256,1024,2048 | 1e-3 | og      | 0.5    | 0.22066      | 0.22514     | 0.22499 |         |

We got the best result on that sheet so far with the mixture of mlp but it could very well be luck. Very close to the best so far. Update : better on kaggle !!!

Let's try with the bigger dataset :

| architecture        | lr   | dataset     | cutoff | eval.py try1 | try2    | try3    | kaggle |
| ------------------- | ---- | ----------- | ------ | ------------ | ------- | ------- | ------ |
| 16,64,256,1024,2048 | 1e-3 | generated 6 | 0      | 0.23007      | 0.21888 | 0.22040 |        |
