##########################################
###     -*- Coding: utf-8 -*-          ###
### Analyste: Charles-Édouard Giguère  ###
###                              .~    ###
###  _\\\\\_                    ~.~    ###
### |  ~ ~  |                 .~~.     ###
### #--O-O--#          ==||  ~~.||     ###
### |   L   |        //  ||_____||     ###
### |  \_/  |        \\  ||     ||     ###
###  \_____/           ==\\_____//     ###
##########################################


require(reticulate, quietly = TRUE, warn.conflicts = FALSE)
setwd("C:/Users/gigc2/Desktop/En_cours/Lacourse/stepmixr/Examples/")
test <- import(module = "test")
py_run_string("for i in range(0,10): print(i)")
test$addone(2)


require(stepmixr)
model1 <- stepmix(n_components = 2, n_steps = 3)
X <- data.frame(x1 = c(0,1,1,1,1,0,0,0,0,0,1,1,0),
                x2 = c(0,1,1,0,0,1,1,0,0,0,1,0,1))
fit1 <- fit(model1, X)
pr1 <- predict(fit1, X)

