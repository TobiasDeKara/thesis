# search_script.R
# This script runs cyclic coordinate descent for L0/L2 regularized least squares,
# as implemented by the 'L0Learn' package, copyright Hussein Hazimeh, 2021.
# This is done using the matrix, penalty coefficients, and number of reserved variables that 
# are written to csv by 'rl_env.py'.

# The csv 'lambdas.csv' has two values: the L0 penalty coefficient and the L2 penalty coefficient, 
# in the 'L0Learn' package these are referred to as 'lambda' and 'gamma'.

library(L0Learn)
options(tidyverse.quiet = TRUE)
library(tidyverse)

x_sub <- read_csv('./param_for_search/x_sub.csv', col_names=TRUE, col_types='d')
col_names <- names(x_sub) %>% as.integer() # these are the indexes of the variables relative to p
# Note that later 'coefs(fit, lambda, gamma)' will give us indexes of the support relative to x_sub

x_sub <- x_sub %>% as.matrix() # L0Learn.fit needs a matrix

y <- read_csv('./param_for_search/y.csv', col_names=FALSE, col_types='d') %>%
	as.matrix()
lambdas <- read_csv('./param_for_search/lambdas.csv', col_names=FALSE, col_types='d')
lambda <- lambdas[[1,1]]
gamma <- lambdas[[2,1]]
len_zub <- read_csv('./param_for_search/len_zub.csv', col_names=FALSE, col_types='i')
len_zub <- len_zub[[1,1]]

fit <- L0Learn.fit(x_sub, y, algorithm="CDPSI", penalty='L0L2', 
	nGamma=1, gammaMin=gamma, gammaMax=gamma, lambdaGrid=list(lambda), excludeFirstK=len_zub)
coefs_with_intercept <- coef(fit, lambda, gamma)
betas_unsorted <- coefs_with_intercept[-1,]
index_of_support_in_x_sub <- coefs_with_intercept@i[-1]
index_of_support_in_p_unsorted <- col_names[index_of_support_in_x_sub]

betas <- betas_unsorted[order(index_of_support_in_p_unsorted)] %>% unname()
support <- index_of_support_in_p_unsorted %>% sort()

write.csv(betas, './results_of_search/betas.csv', row.names=FALSE)
write.csv(support, './results_of_search/support.csv', row.names=FALSE)


