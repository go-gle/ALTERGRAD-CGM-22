"""Gets the parameters (means, variance) to standard scale the condition vector
and saves it in 'cond_standard_scaling

"""

import os
import numpy as np
import pandas as pd
import re
from extract_feats import extract_feats
import torch

from sklearn.preprocessing import StandardScaler

parent_train = '../data/train/description/'  # path to train (idea : try before and after generating graphs)
features_list = []  # will hold all the features
for txt_file in os.listdir(parent_train):
    index = txt_file[6:-4]
    features_list.append([int(index)] + extract_feats(parent_train + txt_file))

df_feats = pd.DataFrame(features_list, columns=['index', 'nodes', 'edges', 'degree', 'triangles', 'clust_coef', 'max_k_core', 'communities'])
df_feats.set_index('index', inplace=True)
df_feats.sort_index(inplace=True)

scaler = StandardScaler()
scaler.fit(df_feats)
scaling = np.concatenate((scaler.mean_.reshape(1, -1), scaler.scale_.reshape(1, -1)), axis=0)
scaling = torch.from_numpy(scaling)
torch.save(scaling, 'cond_standard_scaling')
