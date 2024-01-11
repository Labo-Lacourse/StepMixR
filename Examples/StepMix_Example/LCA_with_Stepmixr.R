### Fichier CEG pour vérifier que tous les codes python fonctionne avec 
### le package StepmixR. 

# # Tutoriel sur le package StepMix (version R)
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

### Premièrement, je m'assure d'avoir la dernière version de stepmixr.
update.packages()
devtools::install_github("Labo-Lacourse/StepMixR@DEV")
setwd("C:/Users/gigc2/Desktop/En cours/Stepmix/StepMixR")

### Appel des librairies. 
library(stepmixr)
library(ggplot2)
library(dplyr, warn.conflicts = FALSE, quiet = TRUE)
theme_set(theme_bw() + theme(legend.position = "bottom"))

# Chargement du fichier de données iris.
# Le fichier est importé du package sklearn.
# L'objet `df` est un data frame pandas (options `as_frame=True`)
# l'objet `target` est aussi un data frame pandas.
df <- iris[, 1:4]; target <- iris[, 5] 


# On peut afficher pour les sépales et pétales séparément
# un graphique qui montre les longueurs sur les largeurs.
# La couleur illustre les trois types de fleurs.
# La séparation des trois espèces est assez claire; l'algorithme
# devrait donc reproduire la structure assez facilement.

# +
options(jupyter.plot_mimetypes = "image/svg+xml",
       repr.plot.height = 4)

iris_by_type <- rbind(data.frame(type = "Sepal",Length = iris[,1], Width = iris[,2], 
                                 species = iris[,5]),
                      data.frame(type = "Petal",Length = iris[,3], Width = iris[,4],
                                 species = iris[,5]))

ggplot(iris_by_type, aes(Length, Width, color = species)) + 
    geom_point(size = 3) + facet_wrap(~type, scales = "free")
# -

# On crée maintenant un modèle stepmix qu'on ajuste et qu'on
# fit au jeu de données iris.

model <- stepmix(n_components=3, measurement="continuous", 
                 random_state=123)
fit1 <- fit(model, df)
fit1

# On sauvegarde maintenant les prédictions de l'appartenance
# au groupe.

df['Class.Membership'] = predict(fit1, df[,1:4])
table(df['Class.Membership'])

# On affiche un bar plot de l'attribution dans chaque classe.

groups_mod1 <- sprintf("group %d (n = %d)", 1:3, table(df['Class.Membership']))
tab_mod1 <- table(df['Class.Membership'])
names(tab_mod1) <- groups_mod1
barplot(tab_mod1, col = c("red", "green", "blue"))

# On peut comparer les espèces de fleur aux classes trouvées selon
# les caractéristiques seulement. On constate que 95% des données
# correspondent aux valeurs originales.

table(df[,'Class.Membership'], target)

# On peut aussi examiner les valeurs des probabilités a posteriori.

post_prob_mod1 = predict_proba(fit1,df[,1:4])

# Pourquoi certaines classes attribuées ne correspondent pas à leur
# classe?
# Affichons les points qui n'ont pas un match parfait avec l'espace.
# On peut voir que la probabilité d'appartenance est plus basse car
# il s'agit de points à la frontière des caractéristiques de deux
# espèces.

post_prob_mod1[df[,'Class.Membership'] != (0:2)[target],] %>% 
    head %>% 
    round(3)


# Les points qui sont classées dans un groupe différent de leur espèce
# d'origine sont identifiés par des carrés. On peut les retrouver à la
# frontière des deux espèces versicolor et virginica.

# +
iris_by_type$Class_different <-  factor(df[,'Class.Membership'] != (0:2)[target], 
                                        labels = c("same", "different"))

ggplot(iris_by_type, aes(Length, Width, color = species, shape = Class_different)) + 
    geom_point(size = 3) + facet_wrap(~type, scales = "free")
# -

# Finalement, on peut voir que le rand score est très élevé.

fossil::rand.index(as.numeric(target), df[,'Class.Membership'])

# # 2. Analyses sur des données dichotomiques.

# Création de données dichotomiques basées sur les quantiles.

# binary_data = sapply(df[,1:4], function(x) (0:1)[cut(x,breaks = 2)])
binary_data <- read.csv("iris_bin.csv")[,2:5]

head(binary_data)
table(binary_data[,2])
table(binary_data[,3])

# On fait le modèle et on ajuste les données au jeux de données
# binaires. On extrait ensuite l'attribution dans les groupes.

model <- stepmix(n_components=3, measurement="binary",
                verbose=1, random_state=123)
fit1 <- fit(model, binary_data)
df['binary_pred'] <- predict(fit1, binary_data)


# On peut voir que le type de fleur n'est pas aussi bien prédit
# et que le rand_score n'est pas aussi bon. Ceci est attendu car
# en dichotomisant on perd de l'information.
length(target)
table(target, df[,'binary_pred'])
sprintf("Rand index = %.4f", fossil::rand.index(as.numeric(target), 
                                                df[,'binary_pred']))

binary_means = fit1$get_parameters()[["measurement"]][["pis"]]

# Les paramètres sont bornés de 0 à 1 car ils représentent la probabilité
# que le feature soit 1 (par rapport à la référence de 0).

data.frame(Prob = as.vector(binary_means),
           group = factor(rep(1:3, 4), labels = c("Group 1", "Group 2", "Group 3")),
           features = rep(1:4, c(3,3,3,3))) %>% 
    ggplot(aes(features,Prob, group = group, col = group)) + 
    geom_line() + geom_point()

# # 3. Analyses sur des données catégorielles.
# on reprend l'exemple continue en séparant en trois catégories
# ordonnées.

#categorical_data = sapply(df[, 1:4], function(x) (0:2)[cut(x,breaks = 3)])
categorical_data <- read.csv("iris_cat.csv")[,2:5]

# À noter que les colonnes sont indexées 0, 1, 2. On ajuste le modèle avec l'option catégorielle, on fait l'ajustement
# du modèle et on prédit l'appartenance au trois catégories.

model = stepmix(n_components=3, measurement="categorical",
                verbose=1, random_state=123)
fit1 <- fit(model, categorical_data)
df[,'categorical_pred'] = predict(fit1, categorical_data)

# On change les catégories de références afin de correspondre aux espèces.

df[, 'categorical_pred2'] <- c(0, 2, 1)[df[, 'categorical_pred'] + 1]


# On peut voir que l'organisation des classes est très bonne. Le rand score
# se situe entre la prédiction en continue et celle dichotomique.

table(target, c(1,3,2)[df[, 'categorical_pred'] + 1])

fossil::rand.index(as.numeric(target), df[,'categorical_pred'])

names(df)

rand_score_models <- data.frame(
    rand_index = c(fossil::rand.index(as.numeric(target), df[,'binary_pred']),
                   fossil::rand.index(as.numeric(target), df[,'categorical_pred']),
                   fossil::rand.index(as.numeric(target), df[,'Class.Membership'])))
rand_score_models$type = c("binary", "Categorical", "Continuous")

ggplot(rand_score_models, aes(type, rand_index)) + geom_point() + geom_line(group = 1, color = "blue")

# # 4. Analyses sur des données de types mélangés

# La fonction suivante permet de faire un modèle utilisant des données
# de plusieurs types différents. L'objet `df_mixed` contient les
# données et l'objet `mixed_descriptor` l'information sur le type associé
# à chaque colonne.

## More complex models need a more complex description
### StepMix provides a function to quickly build mixed descriptors for DataFrames
df_mixed <- data.frame(cbind(df[,c(1,3)], Petal.Width = binary_data[,4], 
                  Sepal.Width = categorical_data[,2])) 
### Unspecified variables are simply not included in mixed_data
md <- mixed_descriptor(data = df_mixed, continuous = c("Sepal.Length", "Petal.Length"), 
                       binary = "Petal.Width", categorical = "Sepal.Width")

# On ajuste le modèle avec deux variables continues, une variable dichotomique
# et une variable catégorielle.

# Pass descriptor to StepMix and fit model
model = stepmix(n_components=3, measurement=md$descriptor, 
                random_state=as.integer(123))
# Fit model
mixed_fit <- fit(model, df_mixed)
mixed_fit

# Class predictions
md$data['mixed_pred'] <- predict(mixed_fit, X = md$data)
table(md$data['mixed_pred'])

# Voici le tableau croisé avec les espèces et le rand score.

table(iris$Species,md$data[,'mixed_pred'])

fossil::rand.index(as.numeric(target), md$data[,'mixed_pred'])

# # 5. Analyses incluant des données manquantes (type = continue)

# Ici, on recopie les données en continue et on génère
# des données manquantes (20 %)

df_na <- as.matrix(df)
df_na[sample(1:600,120)] <- NaN 
df_na <- as.data.frame(df_na)
# Je reprends l'exemple de python pour comparer le résultat. 
df_na <- read.csv("continuous_data_nan.csv", check.names = FALSE)[,2:5]

head(df_na[,1:4])

# Comme toujours, on crée le modèle, on l'ajuste et on fait la prédiction.

model <- stepmix(n_components=3, measurement="continuous_nan",
                verbose=1, random_state=123)
fit1 <- fit(model, df_na[,1:4])
df[,'continuous_pred_nan'] <- predict(fit1, df_na[,1:4])

# On compare les prédictions aux espèces originales.

table(target, df[,'continuous_pred_nan'])

fossil::rand.index(as.numeric(target), df[,'continuous_pred_nan'])

# # 6. Modèle structurelle: Classe + prédictions

# On simule les données selon les modèles présentés dans l'article
# de Zsuzsa Bakk.
#
# - X contient les données sur lesquels on ajuste les classes latentes
# - y est la variable dépendante (composante structurelle).
# - target contient les "vrais" valeurs de groupes.

dbr <- data_bakk_response(n_samples=2000, sep_level=.9, random_state=42)
X <- dbr[[1]]; y <- dbr[[2]]; target <- dbr[[3]]

# On crée le modèle, on ajuste aux données X et Y.
model = stepmix(n_components=3, measurement='binary',
                structural='gaussian_unit', verbose=1,
                random_state=123)

fit1 <- fit(model, X, y)
fit1

# On peut comparer les classes obtenues aux prédictions.

preds = predict(fit1, X, y)
table(preds)
# J'ajoute +1 car R commence ses index à 1.
preds2 = c(0, 2, 1)[preds+1]
table(preds2)

preds_tabs = table(target, preds2)
preds_tabs

preds_pgood = sum(diag(preds_tabs)) / sum(preds_tabs) * 100
sprintf("%.2f %% de bonne prédiction",preds_pgood)


# On s'intéresse aussi à la prédiction de la valeur y.
# Les valeurs représente la moyenne de ces valeurs dans chaque
# groupe.

fit1$get_parameters()['structural']

# Normalement, dans une régression on voudrait avoir des erreurs-types (se)
# et des p-values associées à des tests pour confirmer qu'il y a des
# différence entre les moyennes. Dans stepmix, on peut faire cela à l'aide du
# bootstrapping. Pour l'instant, on ne fait qu'afficher le résultat et
# la méthode de bootstrapping est présenté à la section 11. On peut voir une
# grande séparation entre les moyennes (la séparation dans les données étaient
# de 0.9).

params <- as.table(fit1$get_parameters()[['structural']][['means']][,1])
names(params) <- c("Classe 1", "Classe 2", "Classe 3")

barplot(params, col = c("red", "green", "blue"), ylab = "Moyenne dans la classe")
abline(h = 0, lty = 2)

# # 7. modèle structurel: classes + prédicteurs de classe.

# On simule les données selon les modèles présentés dans l'article de Zsuzsa Bakk.
#
# - X contient les données pour les classes latentes.
# - y est le prédicteur de classe.
# - target are ground truth class memberships

dbr <- data_bakk_covariate(n_samples=2000, sep_level=.9,
                           random_state=42)
X <- apply(dbr[[1]], 2, as.integer); y <- as.integer(dbr[[2]]); target <- dbr[[3]]

# Le modèle avec covariable a une procédure interne dans laquelle on peut
# ajouter des arguments supplémentaires comme le taux d'apprentissage et
# la méthode d'optimisation. Dans cet exemple on utilise l'algorithme de
# Newton-Raphson et on inclue un intercept dans le modèle. Dans R, on doit 
# utiliser une liste plutôt qu'un dictionnaire. Il est important de déclarer 
# le 1 comme un entier pour qu'il soit transféré à Python comme tel.

opt_params = list(
    method    = 'newton-raphson',  # Can also be "gradient",
    intercept = TRUE,
    max_iter  = as.integer(1)  # Number of opt. step each time we update the model
)

# On définit le modèle et on l'ajuste aux données.
model = stepmix(n_components=3, measurement='binary',
                structural='covariate', structural_params=opt_params,
                verbose=1, random_state=123)
# Fit data
# Provide both measurement data X and structural data Y
fit1 <- fit(model, X, y)

# On compare nos classes aux "vrais" classes.

preds = predict(fit1, X, y)
preds2 = c(2, 0, 1)[preds + 1]
table(target, preds2)

# Voici la façon d'extraire les betas et les prédictions de
# probabilités *a priori*. On peut voir qu'il faut faire une tranformation
# pour obtenir un beta normalisé.

BETA = fit1$get_parameters()[['structural']][['beta']]
BETA

# On met la catégorie 1 comme référence.
BETA = BETA - c(1,1,1) %o% BETA[1,]

# Voici comment la prédiction est faite. Je prends seulement les 5
# premières rangées.
# 1. On calcule une matrice de design.

XX = cbind(1, y[1:5])
XX

# 2. On multiplie X par beta.

la = XX %*% t(BETA)
exp(la)

# 3. Pour chaque rangée, on calcule
# $p_i = exp(\lambda_i)/\sum_i exp(\lambda_i)$


PRB_APRIORI = exp(la) / matrix(apply(exp(la), 1, sum), 5, 3)

# # 8. Estimation 1-step, 2-step et 3-step.

# On simule les données selon l'article de Zsuzsa Bakk.
#
# - X contient les données sur lesquels on ajuste les classes latentes
# - y est la variable dépendante (composante structurelle).
# - target contient les "vrais" valeurs de groupes.

dbr = data_bakk_response(n_samples=2000, sep_level=.8, random_state=42)
X <- apply(dbr[[1]], 2, as.integer); y <- as.double(dbr[[2]]) ; target <- dbr[[3]]
y

# Estimation 1 étape (1-step). Ici, tout est mesuré en une étape.
# C'est-à-dire que les paramètres de la classe vont changer selon qu'on
# inclue la variable y ou non.

model = stepmix(n_components=3, measurement='binary', n_steps=1,
                structural='gaussian_unit', verbose=0,
                random_state=123)
fit1 <- fit(model, X, y)
ll_1 = fit1$score(X, y)

# Modèle à 2 étapes. On fait le modèle de classe latente, ensuite on
# prédit y en utilisant les paramètres de classe latente fixée.
#

model$n_steps = as.integer(2)
fit2 <- fit(model,X, y)
ll_2 = fit2$score(X, y)


# Modèle à 3 étapes. On fait le modèle de classe latente, ensuite on
# sauvegarde les données de classe et on fait la prédiction basée sur
# cette classe. Le modèle à trois étapes (three steps) ne fonctionne pas 
# si on envoie un `data.frame`, je dois donc envoyer une matrice plutôt. 

model = stepmix(n_components=3, measurement='binary', n_steps=as.integer(3),
                structural='gaussian_unit', verbose=0,
                random_state=123)
y <- matrix(y, 2000, 1)
fit3 <- fit(model, X, y)
ll_3 <- fit3$score(X, y)

c(sprintf("1-step : %.4f", ll_1),
  sprintf("2-step : %.4f", ll_2),
  sprintf("3-step : %.4f", ll_3))


# Voici comment naviguer à travers les différents paramètres de stepmix.

model = stepmix(n_components=3, measurement='binary', n_steps=3,
                structural='gaussian_unit', verbose=0,
                random_state=123)

# On navigue à travers les différentes options.

result = expand.grid(c = c(NA, "BCH", "ML"), a = c("modal", "soft")) 

# +

for(i in 1:6){
    if(is.na(result[i, "c"]))
        model$correction = NULL
    else
        model$correction = result[i, "c"]
    model$assignment = result[i, "a"]
    fitca <- fit(model, X, y)
    result[i,"ll"] <- fitca$score(X, y)
}

# -

# On met toutes les valeurs de résultats dans un data.frame.

result

# # 9. Comment trouver les paramètres optimaux ?
# Dans cet exemple, on tente de trouver le nombre de composantes et
# le nombre d'étapes optimales?

# On simule à nouveau des données basées sur l'article de Szusza Bakk.

dbr = data_bakk_response(n_samples=2000, sep_level=.7, 
                         random_state=42)
X = dbr[[1]]
y = dbr[[2]]

# On définit le modèle de base.

model = stepmix(n_components=3, n_steps=1, measurement='bernoulli',
                verbose=0, 
                structural='gaussian_unit', random_state=42)
fit1  <- fit(model, X, y)
fit1
# On utilise maintenant la grille de recherche (grid search) de scikit-Learn.
# On teste des classes de 1 à 8 et des estimations à 1, 2 et 3 étapes. 3
# estimations sont faits pour chaque combinaison ce qui donne 72 estimations.
# Ceci suppose que scikit-Learn est installé sur la version de python utilisée 
# dans R. 

mod_sel <- reticulate::py_run_string("from sklearn.model_selection import GridSearchCV, ParameterGrid")
grid = list(n_components = 1:8, n_steps = 1:3)  # une liste R est converti en dict python

# Comme stepmix utilise les standards de sci-kit learn (fit et predict)
# la méthode gridsearch s'utilise facilement.

gs <-  mod_sel$GridSearchCV(estimator=fit1, cv=as.integer(3), 
                            param_grid=grid)

# On peut transmettre le fit automatiquement à tous les modèles
# de la grille de recherche.

gs$fit(X, y)

# On peut extraire les résultats de la grille.

results <- gs$cv_results_
results.df <- data.frame(param_n_components = unlist(results[['param_n_components']]),
                         param_n_steps      = unlist(results[['param_n_steps']]),
                         mean_test_score    = results[['mean_test_score']])

# On affiche les paramètres par nombre de composantes et par nombre d'étapes.
# Plus l'estimateur est élevée, meilleur est l'ajustement.

ggplot(results.df, aes(param_n_components, y = mean_test_score, color = factor(param_n_steps))) + 
    geom_line(aes(group = param_n_steps), lwd = 2) + xlab("Nombre de classes") +
    labs(color = "Nombre d'étapes") + ylab("log-likelihood")


# Same model and grid as above
AIC <- BIC <- NULL
for(g in reticulate::iterate(mod_sel$ParameterGrid(grid))){
    model$n_components = g$n_components
    model$n_steps = g$n_steps
    fitg = fit(model, X, y)
    AIC <- c(AIC, fitg$aic(X, y))
    BIC <- c(BIC, fitg$bic(X, y))
}

# Save results to a dataframe
results.df <- cbind(results.df, AIC, BIC)

# On peut aussi utiliser l'AIC et le BIC qui pénalise respectivement pour le
# nombre de paramètres et pour le nombre de paramètres et la taille
# d'échantillon. Ici le sens est renversé, plus le AIC/BIC est bas,
# plus l'ajustement est bon.

# AIC
ggplot(results.df, aes(x = param_n_components, y = AIC, color = factor(param_n_steps))) + 
  geom_line(mapping = aes(group = param_n_steps), lwd = 2) + 
  xlab("Nombre de classes") +
  labs(color = "Nombre d'étapes") + 
  ylab("Akaike information criterion")


# BIC
ggplot(results.df, aes(param_n_components, y = BIC, 
                       color = factor(param_n_steps))) + 
  geom_line(aes(group = param_n_steps), lwd = 2) + 
  xlab("Nombre de classes") +
  labs(color = "Nombre d'étapes") + 
  ylab("Bayes information criterion")

# # 10. Extraction des paramètres.

# On simule à nouveau des données selon l'article de Szusza Bakk.
dbr <- data_bakk_response(n_samples=2000, sep_level=.7,
                          random_state=42)
X <- dbr[[1]]; y <- matrix(dbr[[2]], 2000, 1); target <- dbr[[3]]

# On ajuste le modèle et on l'ajuste au modèle.

model = stepmix(n_components=3, n_steps=1, measurement='bernoulli',
                structural='gaussian_unit', random_state=42,
                progress_bar = 0)
fit1 <- fit(model, X, y)

# On peut extraire les paramètres.
params = fit1$get_parameters()

# Probabilité *a priori*
params['weights']

# On peut extraire le type de mesure. 'pis' réfère à la probabilité dans
# chaque classe.

# On peut afficher les probabilités
params[['measurement']][['pis']]

# # 11. Apply bootstrap to estimate parameter variances

# On simule à nouveau des données basées sur l'article de Szusza Bakk.
dbr <- data_bakk_response(n_samples=2000, sep_level=.9,
                          random_state=42)
X <- dbr[[1]]; y <- matrix(dbr[[2]], 2000, 1); target <- dbr[[3]]


# On propose un modèle de mesure dichotomique avec variable dépendante
# normale.
model = stepmix(n_components=3, n_steps=1, measurement='bernoulli',
                structural='gaussian_unit', random_state=42, verbose=0,
                max_iter=2000)
fit1 <- fit(model, X, y)

# On effectue 1000 échantillon boostrap de nos valeurs.
bs_params = bootstrap_stats(fit1, X, y, n_repetitions=1000)

# J'extrait les index pour comprendre les paramètres que l'on peut
# extraire:
#    1) Les pis probabilités d'un 1 pour les 6 outcomes par groupe,
#    2) les probabilités d'appartenir à chacun des groupes et
#    3) les moyennes de la variable y dans chaque groupe.

level_header = c('model', 'model_name', 'param', 'class_no', 'variable')
bs_params[['samples']][, level_header] %>% unique

# Les probabilités d'obtenir un 1 dans chaque colonne de X pour
# chaque groupe sont extraites (meanX).
# Il s'agit de la partie mesure qui détermine les groupes.

meanX = bs_params[['samples']] %>% 
  filter(param == 'pis')


# On prépare deux fonctions utiles pour estimer la statistique z (mean/se)
# et la p-value. Ces fonctions serviront à construire les tableaux de
# coefficients.

z <- function(x){
  # Calculate z-statistic: mean/ se.
  mean(x) / sd(x)
}

pval <- function(x){
  # p-value (bilateral)
  round(2*pnorm(-abs(mean(x) / sd(x))), 4)
}

# On estime la moyenne, l'erreur-type, la valeur Z et la p-value associée
# des probabilités d'obtenir un 1 sur les 6 colonnes de X (meanX).

meanX_table <- meanX %>% 
  group_by(class_no, variable) %>% 
  summarize(mean = mean(value), 
            std  = sd(value),
            z = z(value), 
            pval = pval(value), .groups = "drop") %>% 
  as.data.frame

meanX_table

# Pour aider à la compréhension, on peut afficher les moyennes et les se sur
# un graphique.


ggplot(meanX_table, aes(x = variable, y = mean, group = class_no, color = factor(class_no))) + 
  geom_point() + geom_line(lwd = 1.2) + 
  geom_errorbar(aes(ymin = mean - std, ymax = mean + std), width = 0.2) +
  labs(color = "Group")


# J'extrais maintenant les probabilités d'appartenance.

prob = bs_params[['samples']] %>% 
  filter(param == 'class_weights')


prob = bs_params['samples'].xs('class_weights', level=2)[["value"]]
prob_table = prob %>% 
  group_by(class_no) %>% 
  summarize(mean = mean(value), 
            std  = sd(value),
            z = z(value), 
            pval = pval(value), .groups = "drop") %>% 
  as.data.frame

prob_table

# Finalement, on peut extraire les moyennes de y dans chaque groupe.

struct = bs_params[['samples']] %>% 
  filter(model == 'structural')
  
struct %>% 
  group_by(class_no) %>% 
  summarize(mean = mean(value), 
            std  = sd(value),
            z = z(value), 
            pval = pval(value), .groups = "drop") %>% 
  as.data.frame

# Comme tel, il n'y a pas d'intérêt au tableau précédent. On est plutôt
# intéressé à la différence des moyennes entre les groupes.
# 0 vs 1, 0 vs 2 et 1 vs 2.

struct = struct.reset_index()

struct_ctr = data.frame(matrix(nrow = 1000, ncol = 0))
for (i in 0:1){
  for (j in (i+1):2){
    si = struct[struct['class_no'] == i, "value"]
    sj = struct[struct['class_no'] == j, "value"]
    struct_ctr[sprintf('G%d_G%d', i, j)] = sj - si
  }
}

struct_ctrs = tidyr::pivot_longer(struct_ctr, cols = G0_G1:G1_G2, 
                                  names_to = "contrast", values_to = "value")

# On peut voir que les contrastes sont tous significatifs.
# Les groupes discriminent bien la variable y.
struct_ctrs %>% 
  group_by(contrast) %>% 
  summarize(mean = mean(value), 
            std  = sd(value),
            z = z(value), 
            pval = pval(value), .groups = "drop") %>% 
  as.data.frame
