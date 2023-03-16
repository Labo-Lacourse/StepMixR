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

require(dplyr, quietly = TRUE, warn.conflicts = FALSE)
require(ggplot2, quietly = TRUE, warn.conflicts = FALSE)
theme_set(theme_bw() + theme(legend.position = "bottom"))
require(CUFF, quietly = TRUE, warn.conflicts = FALSE)

require(reticulate)
reticulate::use_python("C:/Python39/")
require(phateR)


data(tree.data.small)

# Run PHATE
phate.tree <- phate(tree.data.small$data)
summary(phate.tree)




