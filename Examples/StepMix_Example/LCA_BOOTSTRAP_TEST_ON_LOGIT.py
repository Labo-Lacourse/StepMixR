# -*- coding: utf-8 -*-
"""
Exemple de tests sur les logits par bootstrapping.

Created on Thu Oct 12 14:19:38 2023

@author: gigc2
"""

from stepmix.stepmix import StepMix
from stepmix.datasets import data_bakk_response, data_bakk_covariate
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import rand_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from stepmix.utils import get_mixed_descriptor
from scipy.stats import norm

X, y, _ = data_bakk_response(n_samples=2000, sep_level=.9, random_state=42)

# On propose un modèle de mesure dichotomique avec variable dépendante
# normale.

model = StepMix(n_components=3, n_steps=1, measurement='bernoulli',
                structural='gaussian_unit', random_state=42, verbose=0,
                max_iter=2000)
model.fit(X, y)


model.get_parameters()

# On effectue 1000 échantillon boostrap de nos valeurs.

bs_params = model.bootstrap_stats(X, y, n_repetitions=1000)

# j'extrait les 3 paramètres logit non-normalisés en une seule colonne.

logit_1col = bs_params['samples'].xs("structural").iloc[:, 0]
logit_3cols = logit_1col.values.reshape([3, 1000]).transpose()
logit_3cols = pd.DataFrame(logit_3cols)

# On soustrait la première colonne pour normalisé les résultats. 
logit_3cols_norm = logit_3cols.sub(logit_3cols.iloc[:,0], axis=0)

### Tu peux maintenant faire la moyenne de tes paramêtres. 
mean = logit_3cols_norm.apply(np.mean, axis=0)
se = logit_3cols_norm.apply(np.std, axis=0)

res = pd.DataFrame({"mean": mean[1:3],
                    "se": se[1:3],
                    "z": mean[1:3]/se[1:3],
                    "p-value": 2*norm.cdf(-np.abs(mean[1:3]/se[1:3]))})

res
