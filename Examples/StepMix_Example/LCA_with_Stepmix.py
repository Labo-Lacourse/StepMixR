# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Tutoriel sur le package StepMix
# ## Adapté du [tutoriel de Sacha Morin](https://colab.research.google.com/drive/1KAxcvxjL_vB2lAG9e47we7hrf_2fR1eK?usp=sharing#scrollTo=purSLTpt8TQx)
# ### Charles-Édouard Giguère
# ### 2023-05-25

# # 1. Analyse sur des données continues.

# Dans ce tutoriel, je reproduis les exemples du package stepmix pour
# tester toutes les options du package et vérifier qu'elle fonctionne
# avec stepmixr. J'ai ajouté des lignes de code pour visualiser les
# résultats et voir comment on pourrait utiliser le package en contexte
# pratique.

# ## Importation des packages
# On importe les packages nécessaires pour faire rouler les exemples.

from stepmix.stepmix import StepMix
from stepmix.datasets import data_bakk_response, data_bakk_covariate
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import rand_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from stepmix.utils import get_mixed_descriptor, print_report
from scipy.stats import norm

# Chargement du fichier de données iris.
# Le fichier est importé du package sklearn.
# L'objet `df` est un data frame pandas (options `as_frame=True`)
# l'objet `target` est aussi un data frame pandas.

df, target = load_iris(return_X_y=True, as_frame=True)

# Save actual flower type name
# On peut ajouter le nom des fleurs dans le df qui contient
# les données morphologiques de l'iris.
# Essentiellement, on a la largeur et longueur des pétales et sépales pour
# trois types de fleurs. 50 spécimens pour chaque type (150 spécimens)
df['iris_flower_type'] = target.map({0: 'setosa',
                                     1: 'versicolor',
                                     2: 'virginica'})

# On peut afficher pour les sépales et pétales séparément
# un graphique qui montre les longueurs sur les largeurs.
# La couleur illustre les trois types de fleurs.
# La séparation des trois espèces est assez claire; l'algorithme
# devrait donc reproduire la structure assez facilement.

plt.rcParams['figure.figsize'] = [10, 5]
col_iris = ['red', 'green', 'blue']
species = ['setosa', 'versicolor', 'virginica']
fig, ax = plt.subplots(ncols=2)
for i in np.arange(0, 3):
    ax[0].plot(df[target == i].iloc[:, 0], df[target == i].iloc[:, 1],
               "o", color=col_iris[i], label=species[i])
    ax[1].plot(df[target == i].iloc[:, 1], df[target == i].iloc[:, 2],
               "o", color=col_iris[i], label=species[i])
ax[0].set_title("Sepal")
ax[1].set_title("Petal")
ax[0].legend()
ax[1].legend()
plt.show()

# On crée maintenant un modèle stepmix qu'on ajuste et qu'on
# fit au jeu de données iris.

model = StepMix(n_components=3, measurement="continuous",
                verbose=1, random_state=123)

model.fit(df.iloc[:, 0:4])
print_report(model, df.iloc[:, 0:4])


# On sauvegarde maintenant les prédictions de l'appartenance
# au groupe.

df['Class.Membership'] = model.predict(df.iloc[:, 0:4])
df['Class.Membership'].value_counts()

# On affiche un bar plot de l'attribution dans chaque classe.
tab_mod1 = df['Class.Membership'].value_counts()
groups_mod1 = ['group 1', 'group 2', 'group 3']
groups_mod1 = [i + "(n = {}".format(j) + ")"
               for i, j in zip(groups_mod1, tab_mod1)]
plt.bar(groups_mod1, height=tab_mod1, color=['r', 'g', 'b'])

# On peut comparer les espèces de fleur aux classes trouvées selon
# les caractéristiques seulement. On constate que 95% des données
# correspondent aux valeurs originales.

pd.crosstab(df['Class.Membership'], df['iris_flower_type'])

# On peut aussi examiner les valeurs des probabilités a posteriori.

post_prob_mod1 = model.predict_proba(df.iloc[:, 0:4])

# Pourquoi certaines classes attribuées ne correspondent pas à leur
# classe?
# Affichons les points qui n'ont pas un match parfait avec l'espace.
# On peut voir que la probabilité d'appartenance est plus basse car
# il s'agit de points à la frontière des caractéristiques de deux
# espèces.

post_prob_mod1[df['Class.Membership'] != target, 0:4]


# Les points qui sont classées dans un groupe différent de leur espèce
# d'origine sont identifiés par des carrés. On peut les retrouver à la
# frontière des deux espèces versicolor et virginica.

plt.rcParams['figure.figsize'] = [10, 5]
col_iris = ['red', 'green', 'blue']
species = ['setosa', 'versicolor', 'virginica']
fig, ax = plt.subplots(ncols=2)
for i in np.arange(0, 3):
    df_filtre = df[(target == i) & (target == df['Class.Membership'])]
    df_filtre1 = df[(target == i) & (target != df['Class.Membership'])]
    ax[0].plot(df_filtre.iloc[:, 0], df_filtre.iloc[:, 1],
               "o", color=col_iris[i], label=species[i])
    ax[1].plot(df_filtre.iloc[:, 1], df_filtre.iloc[:, 2],
               "o", color=col_iris[i], label=species[i])
    ax[0].plot(df_filtre1.iloc[:, 0], df_filtre1.iloc[:, 1],
               "s", color=col_iris[i])
    ax[1].plot(df_filtre1.iloc[:, 1], df_filtre1.iloc[:, 2],
               "s", color=col_iris[i])
ax[0].set_title("Sepal")
ax[1].set_title("Petal")
ax[0].legend()
ax[1].legend()
plt.show()

# Finalement, on peut voir que le rand score est très élevé.

rand_score(target, df['Class.Membership'])

# # 2. Analyses sur des données dichotomiques.

# Création de données dichotomiques basées sur les quantiles.

binary_features = []
for c in df.iloc[:, 0:4]:
    c_binary = c.replace("cm", "q=2")
    binary_features.append(c_binary)
    df[c_binary] = pd.qcut(df[c], q=2).cat.codes

# Sélectionner les colonnes and montrer le `data.frame`.

binary_data = df[binary_features]

binary_data.to_csv("iris_bin.csv")
binary_data['sepal length (q=2)'].value_counts()
binary_data.iloc[:, 1].value_counts()
binary_data.iloc[:, 2].value_counts()

# On fait le modèle et on ajuste les données au jeux de données
# binaires. On extrait ensuite l'attribution dans les groupes.

model = StepMix(n_components=3, measurement="binary",
                verbose=1, random_state=123)
model.fit(binary_data)
df['binary_pred'] = model.predict(binary_data)


# On peut voir que le type de fleur n'est pas aussi bien prédit
# et que le rand_score n'est pas aussi bon. Ceci est attendu car
# en dichotomisant on perd de l'information.

pd.crosstab(df['iris_flower_type'], df['binary_pred'])
rand_score(df['iris_flower_type'], df['binary_pred'])

binary_means = model.get_parameters()["measurement"]

# Les paramètres sont bornés de 0 à 1 car ils représentent la probabilité
# que le feature soit 1 (par rapport à la référence de 0).

for i in range(3):
    plt.plot(np.arange(1, 5), binary_means['pis'][i, :], "-o",
             label="group {}".format(i + 1))
plt.xlabel('Features')
plt.ylabel('Probabity of feature in group i')
plt.legend()
plt.show()


# # 3. Analyses sur des données catégorielles.
# on reprend l'exemple continue en séparant en trois catégories
# ordonnées.

categorical_features = []
continuous_data = df.iloc[:, 0:4]
for c in continuous_data:
    # Create new column name
    c_categorical = c.replace("cm", "q=3")
    categorical_features.append(c_categorical)
    df[c_categorical] = pd.qcut(df[c], q=3).cat.codes

# sélection des quatres colonnes catégorielles. À noter que les
# colonnes sont indexées 0, 1, 2

categorical_data = df[categorical_features]
categorical_data
categorical_data.to_csv("iris_cat.csv")


# On ajuste le modèle avec l'option catégorielle, on fait l'ajustement
# du modèle et on prédit l'appartenance au trois catégories.

model = StepMix(n_components=3, measurement="categorical",
                verbose=1, random_state=123)
model.fit(categorical_data)
df['categorical_pred'] = model.predict(categorical_data)

# On change les catégories de références afin de correspondre aux espèces.

df['categorical_pred'] = np.array([0, 2, 1])[df['categorical_pred']]


# On peut voir que l'organisation des classes est très bonne. Le rand score
# se situe entre la prédiction en continue et celle dichotomique.

pd.crosstab(df['iris_flower_type'], df['categorical_pred'])

rand_score_models = np.array(
    [rand_score(df['iris_flower_type'], df['binary_pred']),
     rand_score(df['iris_flower_type'], df['categorical_pred']),
     rand_score(df['iris_flower_type'], df['Class.Membership'])])

plt.plot(['binary', 'Categorical', 'continuous'], rand_score_models,
         ".-")

# # 4. Analyses sur des données de types mélangés

# La fonction suivante permet de faire un modèle utilisant des données
# de plusieurs types différents. Le fichier mixed_data contient les
# données et le fichier mixed_descriptor l'information sur le type associé
# à chaque colonne.

mixed_data, mixed_descriptor = get_mixed_descriptor(
    dataframe=df,
    continuous=['sepal length (cm)', 'petal length (cm)'],
    binary=['petal width (q=2)'],
    categorical=['sepal width (q=3)']
)


# On ajuste le modèle avec deux variables continues, une variable dichotomique
# et une variable catégorielle.
model = StepMix(n_components=3, measurement=mixed_descriptor,
                verbose=1, random_state=123)

model.fit(mixed_data)

df['mixed_pred'] = model.predict(mixed_data)

# Voici le tableau croisé avec les espèces et le rand score.

pd.crosstab(df['iris_flower_type'], df['mixed_pred'])

rand_score(df['iris_flower_type'], df['mixed_pred'])

# # 5. Analyses incluant des données manquantes (type = continue)

# Ici, on recopie les données en continue et on génère
# des données manquantes (20 %)

continuous_data_nan = continuous_data.copy()
for i, c in enumerate(continuous_data_nan.columns):
    continuous_data_nan[c] = continuous_data_nan[c].sample(frac=.8,
                                                           random_state=42 * i)

continuous_data_nan

continuous_data_nan.to_csv("continuous_data_nan.csv")

# Comme toujours, on crée le modèle, on l'ajuste et on fait la prédiction.

model = StepMix(n_components=3, measurement="continuous_nan",
                verbose=1, random_state=123)
model.fit(continuous_data_nan)
df['continuous_pred_nan'] = model.predict(continuous_data_nan)

# On compare les prédictions aux espèces originales.

pd.crosstab(df['iris_flower_type'], df['continuous_pred_nan'])

rand_score(df['iris_flower_type'], df['continuous_pred_nan'])

# # 6. Modèle structurelle: Classe + prédictions

# On simule les données selon les modèles présentés dans l'article
# de Zsuzsa Bakk.
#
# - X contient les données sur lesquels on ajuste les classes latentes
# - y est la variable dépendante (composante structurelle).
# - target contient les "vrais" valeurs de groupes.

X, y, target = data_bakk_response(n_samples=2000,
                                  sep_level=.9, random_state=42)

# On crée le modèle, on ajuste aux données X et Y.
model = StepMix(n_components=3, measurement='binary',
                structural='gaussian_unit', verbose=1,
                random_state=123)

model.fit(X, y)

# On peut comparer les classes obtenues aux prédictions.

preds = model.predict(X, y)
pd.Series(preds).value_counts()
preds2 = np.array([0, 2, 1])[preds]
pd.Series(preds2).value_counts()

preds_ctabs = pd.crosstab(target, preds2)
preds_ctabs

preds_pgood = sum(np.diag(preds_ctabs)) / sum(sum(preds_ctabs.values)) * 100
print("{} % de bonne prédiction".format(preds_pgood))


# On s'intéresse aussi à la prédiction de la valeur y.
# Les valeurs représente la moyenne de ces valeurs dans chaque
# groupe.

model.get_parameters()['structural']

# Normalement, dans une régression on voudrait avoir des erreurs-types (se)
# et des p-values associées à des tests pour confirmer qu'il y a des
# différence entre les moyennes. Dans stepmix, on peut faire cela à l'aide du
# bootstrapping. Pour l'instant, on ne fait qu'afficher le résultat et
# la méthode de bootstrapping est présenté à la section 11. On peut voir une
# grande séparation entre les moyennes (la séparation dans les données étaient
# de 0.9).

plt.bar(['Classe 1', 'Classe 2', 'Classe 3'],
        height=model.get_parameters()['structural']['means'][:, 0])
plt.plot([-1, 3], [0, 0], "--")
plt.ylabel("Moyennes de y")

# # 7. modèle structurel: classes + prédicteurs de classe.

# On simule les données selon les modèles présentés dans l'article
# de Zsuzsa Bakk.
# X contient les données pour les classes latentes.
# y est le prédicteur de classe.
# target are ground truth class memberships
X, y, target = data_bakk_covariate(n_samples=2000, sep_level=.9,
                                   random_state=42)

X
# Le modèle avec covariable a une procédure interne dans laquelle on peut
# ajouter des arguments supplémentaires comme le taux d'apprentissage et
# la méthode d'optimisation. Dans cet exemple on utilise l'algorithme de
# Newton-Raphson et on inclue un intercept dans le modèle.

opt_params = {
    'method': 'newton-raphson',  # Can also be "gradient",
    'intercept': True,
    'max_iter': 1,  # Number of opt. step each time we update the model
}

# On définit le modèle et on l'ajuste au données.

model = StepMix(n_components=3, measurement='binary',
                structural='covariate', structural_params=opt_params,
                verbose=1, random_state=123)
# Fit data
# Provide both measurement data X and structural data Y
model.fit(X, y)

# On compare nos classes aux "vrais" classes.

preds = model.predict(X, y)
preds2 = np.array([2, 0, 1])[preds]
pd.crosstab(target, preds2)

# Voici la façon d'extraire les betas et les prédictions de
# probabilités *a priori*. On peut voir qu'il faut faire une tranformation
# pour obtenir un beta normalisé.

BETA = model.get_parameters()['structural']['beta']
# On met la catégorie 1 comme référence.
BETA -= BETA[0, :].reshape((1, -1))
BETA

# Voici comment la prédiction est faite. Je prends seulement les 5
# premières rangées.
# 1. On calcule une matrice de design.

XX = np.column_stack((np.repeat(1, 5), y[0:5]))
XX

# 2. On multiplie X par beta.

la = np.dot(XX, BETA.transpose())

# 3. Pour chaque rangée, on calcule
# $p_i = exp(\lambda_i)/\sum_i exp(\lambda_i)$

PRB_APRIORI = np.exp(la) / np.exp(la).sum(axis=1).repeat(3).reshape([5, 3])
PRB_APRIORI

# # 8. Estimation 1-step, 2-step et 3-step.

# On simule les données selon l'article de Zsuzsa Bakk.
#
# - X contient les données sur lesquels on ajuste les classes latentes
# - y est la variable dépendante (composante structurelle).
# - target contient les "vrais" valeurs de groupes.

X, y, target = data_bakk_response(n_samples=2000, sep_level=.8,
                                  random_state=42)
y
# Estimation 1 étape (1-step). Ici, tout est mesuré en une étape.
# C'est-à-dire que les paramètres de la classe vont changer selon qu'on
# inclue la variable y ou non.

model = StepMix(n_components=3, measurement='binary', n_steps=1,
                structural='gaussian_unit', verbose=0,
                random_state=123)

model.fit(X, y)
ll_1 = model.score(X, y)

# Modèle à 2 étapes. On fait le modèle de classe latente, ensuite on
# prédit y en utilisant les paramètres de classe latente fixée.


model.set_params(n_steps=2)
model.fit(X, y)
ll_2 = model.score(X, y)


# Modèle à 3 étapes. On fait le modèle de classe latente, ensuite on
# sauvegarde les données de classe et on fait la prédiction basée sur
# cette classe.

model.set_params(n_steps=3)
model.fit(X, y)
ll_3 = model.score(X, y)

print(f"1-step : {ll_1:.4f}")
print(f"2-step : {ll_2:.4f}")
print(f"3-step : {ll_3:.4f}")


# Voici comment naviguer à travers les différents paramètres de stepmix.

model = StepMix(n_components=3, measurement='binary', n_steps=3,
                structural='gaussian_unit', verbose=0,
                random_state=123)

# On navigue à travers les différentes options.

result = dict(correction=[], assignment=[], log_likelihood=[])

for c in [None, 'BCH', 'ML']:
    for a in ['modal', 'soft']:
        model.set_params(correction=c, assignment=a)
        model.fit(X, y)
        ll = model.score(X, y)
        result['correction'].append(c)
        result['assignment'].append(a)
        result['log_likelihood'].append(ll)

# On met toutes les valeurs de résultats dans un data.frame.

result = pd.DataFrame(result)
result

# # 9. Comment trouver les paramètres optimaux ?
# Dans cet exemple, on tente de trouver le nombre de composantes et
# le nombre d'étapes optimales?

# On simule à nouveau des données basées sur l'article de Szusza Bakk.

X, y, _ = data_bakk_response(n_samples=2000, sep_level=.7,
                             random_state=42)
# On définie le modèle de base.

model = StepMix(n_components=2, n_steps=1, measurement='bernoulli',
                verbose=0,
                structural='gaussian_unit', random_state=42)

model.fit(X, y)
print_report(model, X, y)

# On utilise maintenant la grille de recherche (grid search) de scikit-Learn.
# On teste des classes de 1 à 8 et des estimations à 1, 2 et 3 étapes. 3
# estimations sont faits pour chaque combinaison ce qui donne 72 estimations.

grid = {
    'n_components': [1, 2, 3, 4, 5, 6, 7, 8],
    'n_steps': [1, 2, 3]
}

# Comme stepmix utilise les standards de sci-kit learn (fit et predict)
# la méthode gridsearch s'utilise facilement.

gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)

# On peut transmettre le fit automatiquement à tous les modèles
# de la grille de recherche.

gs.fit(X, y)

# On peut extraire les résultats de la grille.

results = pd.DataFrame(gs.cv_results_)
results["Val. Log Likelihood"] = results['mean_test_score']
results
# On affiche les paramètres par nombre de composantes et par nombre d'étapes.
# Plus l'estimateur est élevée, meilleur est l'ajustement.

sns.set_style("darkgrid")
sns.lineplot(data=results, x='param_n_components', y='Val. Log Likelihood',
             hue='param_n_steps', palette='Dark2')

results = dict(param_n_steps=[], param_n_components=[], aic=[], bic=[])

# Same model and grid as above
model.fit(X, y)


for g in ParameterGrid(grid):
    model = StepMix(n_components=g['n_components'],
                    n_steps=['n_steps'],
                    measurement='bernoulli',
                    verbose=0, structural='gaussian_unit',
                    random_state=42)
    model.set_params(**g)
    model.fit(X, y)
    results['param_n_steps'].append(g['n_steps'])
    results['param_n_components'].append(g['n_components'])
    results['aic'].append(model.aic(X, y))
    results['bic'].append(model.bic(X, y))

# Save results to a dataframe
results = pd.DataFrame(results)

# On peut aussi utiliser l'AIC et le BIC qui pénalise respectivement pour le
# nombre de paramètres et pour le nombre de paramètres et la taille
# d'échantillon. Ici le sens est renversé, plus le AIC/BIC est bas,
# plus l'ajustement est bon.

# AIC
sns.lineplot(data=results, x='param_n_components', y='aic',
             hue='param_n_steps', palette='Dark2')

# BIC
sns.lineplot(data=results, x='param_n_components', y='bic',
             hue='param_n_steps', palette='Dark2')


# # 10. Extraction des paramètres.

# On simule à nouveau des données selon l'article de Szusza Bakk.

X, y, _ = data_bakk_response(n_samples=2000, sep_level=.7,
                             random_state=42)

# On ajuste le modèle et on l'ajuste au modèle.

model = StepMix(n_components=3, n_steps=1, measurement='bernoulli',
                structural='gaussian_unit', random_state=42)
model.fit(X, y)

# On peut extraire les paramètres.
params = model.get_parameters()

# Probabilité *a priori* d'appartenir à chacun des trois groupes.
params['weights']

# On peut extraire le type de mesure. 'pis' réfère à la probabilité dans
# chaque classe. 
params['measurement'].keys()

# On peut afficher les probabilités
# Les colonnes réfèrent aux 6 outcomes binaires et les rangées aux 3 groupes.
params['measurement']['pis']

# # 11. Apply bootstrap to estimate parameter variances

# On simule à nouveau des données basées sur l'article de Szusza Bakk.
X, y, _ = data_bakk_response(n_samples=2000, sep_level=.9,
                             random_state=42)

# On propose un modèle de mesure dichotomique avec variable dépendante
# normale.
model = StepMix(n_components=3, n_steps=1, measurement='bernoulli',
                structural='gaussian_unit', random_state=42, verbose=0,
                max_iter=2000)
model.fit(X, y)
model.get_parameters()


# On effectue 1000 échantillon boostrap de nos valeurs.
bs_params = model.bootstrap_stats(X, y, n_repetitions=1000)
test = model.bootstrap_stats(X, y, n_repetitions=10, progress_bar=False)
# J'extrait les index pour comprendre les paramètres que l'on peut
# extraire:
#    1) Les pis probabilités d'un 1 pour les 6 outcomes par groupe,
#    2) les probabilités d'appartenir à chacun des groupes et
#    3) les moyennes de la variable y dans chaque groupe.

level_header = ['model', 'model_name', 'param', 'class_no', 'variable']
bs_params['samples'].sort_index(level=level_header).index.unique()

# Les probabilités d'obtenir un 1 dans chaque colonne de X pour
# chaque groupe sont extraites (meanX).
# Il s'agit de la partie mesure qui détermine les groupes.

meanX = bs_params['samples'].xs('pis', level=2)[["value"]]


# On prépare deux fonctions utiles pour estimer la statistique z (mean/se)
# et la p-value. Ces fonctions serviront à construire les tableaux de
# coefficients.

def z(x):
    """Calculate z-statistic : mean / se."""
    return x.mean() / x.std()


def pval(x):
    """Calculate the p-value."""
    return np.round(2*norm.cdf(-np.abs(x.mean() / x.std())), 4)


# On estime la moyenne, l'erreur-type, la valeur Z et la p-value associée
# des probabilités d'obtenir un 1 sur les 6 colonnes de X (meanX).

meanX_table = meanX.groupby(['class_no', 'variable']).agg(
    {'value': ['mean', 'std', z, pval]})
meanX_table

# Pour aider à la compréhension, on peut afficher les moyennes et les se sur
# un graphique.

meanX_table = meanX_table.reset_index()
fig, ax = plt.subplots(figsize=(9, 5))
for i in [0, 1, 2]:
    mi = meanX_table[meanX_table['class_no'] == i]
    plt.errorbar(mi['variable'], mi.value['mean'], mi.value['std'],
                 label='Grp ' + str(i))
plt.legend()
plt.show()


# J'extrais maintenant les probabilités d'appartenance.

prob = bs_params['samples'].xs('class_weights', level=2)[["value"]]
prob_table = prob.groupby(['class_no']).agg(
    {'value': ['mean', 'std', z, pval]})
prob_table

# Finalement, on peut extraire les moyennes de y dans chaque groupe.
struct = bs_params['samples'].xs('structural')[["value"]]
struct.groupby(['class_no']).agg(
    {'value': ['mean', 'std', z, pval]})

# Comme tel, il n'y a pas d'intérêt au tableau précédent. On est plutôt
# intéressé à la différence des moyennes entre les groupes.
# 0 vs 1, 0 vs 2 et 1 vs 2.

struct = struct.reset_index()

struct_ctr = pd.DataFrame()

for i in np.arange(0, 2):
    for j in np.arange(i+1, 3):
        si = struct[struct['class_no'] == i][['value']].reset_index(drop=True)
        sj = struct[struct['class_no'] == j][['value']].reset_index(drop=True)
        struct_ctr['G{}_G{}'.format(i, j)] = sj - si

struct_ctrs = pd.DataFrame({"value": struct_ctr.stack()})

# On peut voir que les contrastes sont tous significatifs.
# Les groupes discriminent bien la variable y.
struct_ctrs.groupby(level=1).agg(
    {'value': ['mean', 'std', z, pval]})
