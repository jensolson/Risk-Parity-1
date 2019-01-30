# Risk Parity Helper Calcs
Helper functions to calculate optimal risk parity portfolio weights given historical returns

Dependencies: pandas, numpy, sklearn

* "min_var" finds global minimum variance portfolio
* "NewtonERC" class is the most robust in this file. Finds portfolio weights such that each security contributes equally to portfolio risk
* "get_M2", "get_M3" through "high_moment_F" are a set of functions to find portfolio weights optimized for higher order moments of return distribution
* "getDiversifiedWeights" is a function to be minimized to return portfolio weights to maximize the number of orthogonal bets taken given a number of securities
* "getPCAWeights" calculates portfolio weights after running PCA on return distribution, so that each principal component carries equal loading

