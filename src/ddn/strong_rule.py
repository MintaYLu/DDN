import numpy as np
from sklearn.linear_model import Lasso


def strong_rule(X, y, lambda1):
    n = len(y)
    lambdas = np.abs(np.matmul(X.T, y)) / n
    lambdamax = max(lambdas)
    idx = np.array(
        [i for i in range(X.shape[1]) if lambdas[i] >= 2 * lambda1 - lambdamax]
    )
    drop = X.shape[1] - len(idx)
    # print(X.shape[1], drop)
    return idx


def lasso(y, X, lambda1, beta_in, tol=1e-6, use_strong_rule=True, use_warm=False):
    # select feature index
    n_pred = X.shape[1]
    idx = np.arange(n_pred)

    if use_strong_rule:
        idx = strong_rule(X, y, lambda1)
        if len(idx) == 0:
            return np.zeros(n_pred)
        X = X[:, idx]

    # calculate beta with sklearn lasso
    if use_warm:
        clf = Lasso(
            alpha=lambda1,
            # max_iter=1000,
            # tol=tol,
            fit_intercept=False,
            warm_start=True,
        )
        clf.coef_ = beta_in
        # clf.coef_ = np.zeros((n_pred, ))
    else:
        clf = Lasso(
            alpha=lambda1,
            # max_iter=1000,
            # tol=tol,
            fit_intercept=False,
            warm_start=False,
        )
        clf.coef_ = np.zeros((n_pred,))
    clf.fit(X, y)

    beta = np.zeros(n_pred)
    # beta = np.zeros((len(idx),))
    beta[idx] = clf.coef_

    return beta
