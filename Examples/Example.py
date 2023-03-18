# -*- Coding: utf-8 -*-
"""
Example of the use of stepmix.

We used the mode to identify the three species or IRIS using the 4 features.
@author: Charles-Édouard Giguère
"""

from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from stepmix.stepmix import StepMix

# Import Iris datasets.
iris = datasets.load_iris().data
iris = pd.DataFrame(iris)
species = datasets.load_iris().target

# Print the first three lines.
iris

# Plot the Sepals and Petal with pyplot
fig, ax = plt.subplots(1, 2)
ax[0].scatter(iris.iloc[:, 0], iris.iloc[:, 1], c=species+1)
ax[0].set_xlabel("Sepal Length")
ax[0].set_ylabel("Sepal Width")
ax[1].scatter(iris.iloc[:, 2], iris.iloc[:, 3], c=species)
ax[1].set_xlabel("Petal Length")
ax[1].set_ylabel("Petal Width")

# Now we use the stepmix package.
model = StepMix(n_components=3, n_steps=1,
                measurement="gaussian_full", random_state=8546)
model.fit(iris)
pr = model.predict(iris)

# we change arbitrary label to fit iris species.
is12 = np.any([pr == 1, pr == 2], axis=0)

pd.crosstab(pr, species)

# Can we do better? Probably
model2 = StepMix(n_components=3, n_steps=1,
                 measurement="gaussian_diag", random_state=5648)
model2.fit(iris)
pr2 = model2.predict(iris)
# Now, there are only 10 out of 150 (7%) that are misidentified.
pd.crosstab(pr2, species)

# Let's do the graph we did earlier by identifying those
# misidentified data points.
col = pd.Series(["#440154FF", "#21908CFF", "#FDE725FF"])
fig, ax = plt.subplots(1, 2)
ax[0].scatter(iris.iloc[pr2 == species, 0],
              iris.iloc[pr2 == species, 1], c=species[pr2 == species],
              marker='o')
ax[0].scatter(iris.iloc[pr2 != species, 0],
              iris.iloc[pr2 != species, 1], color=col[species[pr2 != species]],
              marker='x',)
ax[0].set_xlabel("Sepal Length")
ax[0].set_ylabel("Sepal Width")
ax[1].scatter(iris.iloc[:, 2], iris.iloc[:, 3], c=species)
ax[1].set_xlabel("Petal Length")
ax[1].set_ylabel("Petal Width")

# We can see that the two last species can have sepals of the same
# dimension so it is not discriminating very well. Let's try a model
# using only the petal.
model3 = StepMix(n_components=3, n_steps=1,
                 measurement="gaussian_diag", random_state=1234)
model3.fit(iris.iloc[:, 2:4])
pr3 = model3.predict(iris.iloc[:, 2:4])
pr3 = pd.Series([1, 0, 2])[pr3]
# Now, there are only 6 out of 150 (4%) that are misidentified.
pd.crosstab(pr3, species)

# Now suppose that we have two other variables that are related to the
# outcome: y1 (dichotomous) and y2 (continuous).
Y = pd.DataFrame(np.zeros([150, 2]), columns=['y1', 'y2'])
# x1 is binom(1,p = 0.25) in group 1 and 2 and binom(1, p = 0.50)
# in group 3.
p = np.zeros(150)
np.random.seed(3948)
for i in range(0, 150):
    p[i] = 0.25 + (0.25) * (species[i] == 2)
mu = np.array([np.repeat(1, 50),
               np.repeat(0.8, 50),
               np.repeat(1.5, 50)]).reshape(150)
Y.y1 = np.random.binomial(1, p, 150)
Y.y2 = np.random.normal(loc=mu, scale=np.repeat(1, 150), size=150)

# It's the final model...
model4 = StepMix(n_components=3, n_steps=3,
                 measurement="gaussian_diag",
                 structural="gaussian_unit",
                 random_state=1234)
model4.fit(X=iris.iloc[:, 2:4], Y=np.array(Y))
pr4 = model4.predict(X=iris.iloc[:, 2:4], Y=Y)
labels = pd.Series([1, 0, 2])
pr4 = labels[pr4]
pd.crosstab(pr4, species)
model4.get_parameters()
