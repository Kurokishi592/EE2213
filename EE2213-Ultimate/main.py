from EnterMetrics import EnterMetrics
from LinearRegression import linear_regression
from PolynomialRegression import polynomial_regression
from PolyRegressionClassification import polynomial_regression_with_classification
from RidgePolynomialRegression import ridge_poly_regression
from RidgeRegression import ridge_regression
from BinomialLogicticRegression import logistic_regression
from MultinomialLogisticRegression import multinomial_logistic_regression
from OneHotLinearClassification import onehot_linearclassification
from pearson_correlation import pearson_correlation
import numpy as np
from sklearn.metrics import mean_squared_error
from EnterMetrics import EnterMetrics

# no need to add column of 1s to X for regression. X_fitted does it already. First row of w is w0.
X=np.array(
    [[0.5, 1.2, -0.3],
     [-1, 0.8, 1.5],
     [2.3, -0.7, 0.5],
     [0, 1.5, -1.0]
   ]
);

# for one hot encoding (e.g. for multinomial logistic regression), simply type the class number 1, 2, 3...etc
# for binary logistic regression, assign label 1 to the class in interest
# the code will convert to onehot encoding internally
Y=np.array(
    [[1],
     [2],
     [3],
     [1]
     ]
);

# same dont add one column of 1s to X_test for regression
X_test=np.array(
    [[7, 10, 1]
    ]
)



X_fitted=np.hstack((np.ones((len(X),1)),X))
X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))
# linear_regression(X_fitted,Y, X_test_fitted)
# polynomial_regression(X,Y,order=3,X_test=X_test) #order=1 is linear regression
# polynomial_regression_with_classification(X, Y, order=1, X_test=X_test) #use this if linear regression with classification needed
# ridge_regression(X_fitted,Y,LAMBDA=0.1, X_test=X_test_fitted, form='auto') #linear model
# ridge_poly_regression(X, Y, LAMBDA=1, order=2, form='auto', X_test=X_test)

w_initial = np.array(
    [[0, 0, 0], 
     [0.01, -0.02, 0.03],
     [0.05, 0.04, -0.01],
     [-0.03, 0.02, 0.01]
    ]
)
# logistic_regression(X, Y, X_test, w_initial, learning_rate=0.1, num_iters=10000)
multinomial_logistic_regression(X, Y, X_test, w_initial, learning_rate=0.1, num_iters=10000)



#adding one for linear NOT FOR POLYNOMIAL5,-6

# onehot_linearclassification(X_fitted,Y,X_test_fitted)

# pearson_correlation(X,Y)

# print("did you add offset for X if you are using linear regression? and DON'T use offset for polynomial Regression!" )
