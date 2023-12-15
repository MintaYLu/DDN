import numpy as np
import numba


def bcd_org_old(
    beta_in, y, X, lambda1, lambda2, n1=1, n2=1, threshold=1e-6, max_iter=100000
):
    # total feature size must be even
    if X.shape[1] == 0 or X.shape[1] % 2 == 1:
        return []

    # feature size (gene size) for each group
    # x1 (control): feature 0 to p-1
    # x2 (case): feature p to 2*p-1
    p = X.shape[1] // 2

    # initialize beta
    beta = np.copy(beta_in)
    # beta = np.zeros(2 * p)

    if p == 1:
        rho1 = np.sum(y * X[:, 0]) / n1
        rho2 = np.sum(y * X[:, 1]) / n2

        beta2d = solve2d(rho1, rho2, lambda1, lambda2)
        beta[0] = beta2d[0]
        beta[1] = beta2d[1]

        return beta
    else:
        r = 0
        while True:
            beta_old = np.copy(beta)

            for k in range(p):
                x1 = X[:, k]
                x2 = X[:, k + p]
                r = r + 1

                # beta_temp = np.copy(beta)
                # beta_temp[k] = 0
                # beta_temp[k + p] = 0
                # y_residual0 = y - np.dot(X, beta_temp)

                idx = [i for i in range(2 * p) if i not in (k, k + p)]
                y_residual = y - np.dot(X[:, idx], beta[idx])

                # print(np.sum(np.abs(y_residual - y_residual0)))

                rho1 = np.sum(y_residual * x1) / n1
                rho2 = np.sum(y_residual * x2) / n2

                if np.abs(rho1) > 1000 or np.abs(rho2) > 1000:
                    print(rho1, rho2)

                beta2d = solve2d(rho1, rho2, lambda1, lambda2)

                beta[k] = beta2d[0]
                beta[k + p] = beta2d[1]

            betaerr = np.mean(np.abs(beta - beta_old))
            if (betaerr < threshold) or (r > max_iter):
                break

        return beta, r, betaerr


@numba.njit
def bcd_org(beta_in, y, X, lambda1, lambda2, n1=1, n2=1, threshold=1e-6, max_iter=100000):
    # feature size (gene size) for each group
    # x1 (control): feature 0 to p-1
    # x2 (case): feature p to 2*p-1
    p = X.shape[1] // 2

    # initialize beta
    beta = np.copy(beta_in)
    r = 0
    while True:
        beta_old = np.copy(beta)

        for k in range(p):
            x1 = X[:, k]
            x2 = X[:, k + p]
            r = r + 1

            beta_temp = np.copy(beta)
            beta_temp[k] = 0
            beta_temp[k + p] = 0
            y_residual = y - np.dot(X, beta_temp)

            rho1 = np.sum(y_residual * x1) / n1
            rho2 = np.sum(y_residual * x2) / n2

            beta2d = solve2d(rho1, rho2, lambda1, lambda2)

            beta[k] = beta2d[0]
            beta[k + p] = beta2d[1]

        betaerr = np.mean(np.abs(beta - beta_old))
        if (betaerr < threshold) or (r > max_iter):
            break

    return beta, r, betaerr


@numba.njit
def bcd_residual(
    beta_in,
    X1,
    X2,
    y1_resi,
    y2_resi,
    CurrIdx,
    lambda1,
    lambda2,
    threshold,
    max_iter=10000,
):
    p = X1.shape[1]
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    # n_mean = (n1+n2/2) # BUG

    beta1 = np.copy(beta_in[:p])
    beta2 = np.copy(beta_in[p:])

    betaerr_array = np.zeros(2 * p)
    beta = np.zeros(2 * p)
    # betaerr = 100.0

    r = 0
    k_last = CurrIdx
    while True:
        beta1_old = np.copy(beta1)
        beta2_old = np.copy(beta2)

        # if 1:
        #     if r > 0:
        #         b_norm_old = np.sum(np.abs(beta1_old)) + np.sum(np.abs(beta2_old))
        #         err1 = beta1 - beta1_old
        #         err2 = beta2 - beta2_old
        #         b_norm_dif = np.sum(np.abs(err1)) + np.sum(np.abs(err2))
        #         betaerr = b_norm_dif / b_norm_old
        #         if (betaerr < threshold) or (r > max_iter):
        #             break

        for i in range(p):
            if i == CurrIdx:
                continue

            r = r + 1
            k = i

            y1_resi = y1_resi - beta1[k_last] * X1[:, k_last] + beta1[k] * X1[:, k]
            y2_resi = y2_resi - beta2[k_last] * X2[:, k_last] + beta2[k] * X2[:, k]
            rho1 = np.sum(y1_resi * X1[:, k]) / n1
            rho2 = np.sum(y2_resi * X2[:, k]) / n2
            # rho1 = np.sum(y1_resi * X1[:, k]) / n_mean # BUG
            # rho2 = np.sum(y2_resi * X2[:, k]) / n_mean # BUG

            beta2d = solve2d(rho1, rho2, lambda1, lambda2)
            beta1[k] = beta2d[0]
            beta2[k] = beta2d[1]

            k_last = k

        betaerr_array[:p] = beta1 - beta1_old
        betaerr_array[p:] = beta2 - beta2_old
        betaerr = np.mean(np.abs(betaerr_array))

        if (betaerr < threshold) or (r > max_iter):
            break

    beta[:p] = beta1
    beta[p:] = beta2
    return beta, r, betaerr


@numba.njit
def bcd_corr(
    beta_in,
    cur_node,
    lambda1,
    lambda2,
    corr_matrix_1,
    corr_matrix_2,
    threshold=1e-6,
    max_iter=100000,
):
    p = int(len(beta_in) / 2)
    beta1 = np.copy(beta_in[:p])
    beta2 = np.copy(beta_in[p:])
    beta_dif = np.zeros(2 * p)

    r = 0
    for _ in range(max_iter):
        beta1_old = np.copy(beta1)
        beta2_old = np.copy(beta2)

        for k in range(p):
            if k == cur_node:
                continue
            r = r + 1

            betaBar1 = -beta1
            betaBar2 = -beta2
            betaBar1[k] = 0
            betaBar2[k] = 0
            betaBar1[cur_node] = 1
            betaBar2[cur_node] = 1

            rho1 = np.sum(betaBar1 * corr_matrix_1[:, k])
            rho2 = np.sum(betaBar2 * corr_matrix_2[:, k])
            # rho1 = betaBar1 @ corr_matrix_1[:, k]
            # rho2 = betaBar2 @ corr_matrix_2[:, k]

            beta2d = solve2d(rho1, rho2, lambda1, lambda2)
            beta1[k] = beta2d[0]
            beta2[k] = beta2d[1]

        beta_dif[:p] = beta1 - beta1_old
        beta_dif[p:] = beta2 - beta2_old
        delta_beta = np.mean(np.abs(beta_dif))

        if delta_beta < threshold:
            break

        # print(delta_beta)

    beta = np.zeros(2 * p)
    beta[:p] = beta1
    beta[p:] = beta2

    return beta, r, delta_beta


@numba.njit
def solve2d(rho1, rho2, lambda1, lambda2):
    # initialize output
    area_index = 0
    beta1 = 0
    beta2 = 0

    if (
        rho2 <= (rho1 + 2 * lambda2)
        and rho2 >= (rho2 - 2 * lambda2)
        and rho2 >= (2 * lambda1 - rho1)
    ):
        area_index = 1
        beta1 = (rho1 + rho2) / 2 - lambda1
        beta2 = (rho1 + rho2) / 2 - lambda1
    if rho2 > (rho1 + 2 * lambda2) and rho1 >= (lambda1 - lambda2):
        area_index = 2
        beta1 = rho1 - lambda1 + lambda2
        beta2 = rho2 - lambda1 - lambda2
    if (
        rho1 < (lambda1 - lambda2)
        and rho1 >= -(lambda1 + lambda2)
        and rho2 >= (lambda1 + lambda2)
    ):
        area_index = 3
        beta1 = 0
        beta2 = rho2 - lambda1 - lambda2
    if rho1 < -(lambda1 + lambda2) and rho2 >= (lambda1 + lambda2):
        area_index = 4
        beta1 = rho1 + lambda1 + lambda2
        beta2 = rho2 - lambda1 - lambda2
    if (
        rho1 < -(lambda1 + lambda2)
        and rho2 < (lambda1 + lambda2)
        and rho2 >= -(lambda1 + lambda2)
    ):
        area_index = 5
        beta1 = rho1 + lambda1 + lambda2
        beta2 = 0
    if rho2 < -(lambda1 - lambda2) and rho2 >= (rho1 + 2 * lambda2):
        area_index = 6
        beta1 = rho1 + lambda1 + lambda2
        beta2 = rho2 + lambda1 - lambda2
    if (
        rho2 >= (rho1 - 2 * lambda2)
        and rho2 < (rho1 + 2 * lambda2)
        and rho2 <= (-2 * lambda1 - rho1)
    ):
        area_index = 7
        beta1 = (rho1 + rho2) / 2 + lambda1
        beta2 = (rho1 + rho2) / 2 + lambda1
    if rho2 < (rho1 - 2 * lambda2) and rho1 <= -(lambda1 - lambda2):
        area_index = 8
        beta1 = rho1 + lambda1 - lambda2
        beta2 = rho2 + lambda1 + lambda2
    if (
        rho1 <= (lambda1 + lambda2)
        and rho1 >= -(lambda1 - lambda2)
        and rho2 <= -(lambda1 + lambda2)
    ):
        area_index = 9
        beta1 = 0
        beta2 = rho2 + lambda1 + lambda2
    if rho1 > (lambda1 + lambda2) and rho2 <= -(lambda1 + lambda2):
        area_index = 10
        beta1 = rho1 - lambda1 - lambda2
        beta2 = rho2 + lambda1 + lambda2
    if (
        rho2 > -(lambda1 + lambda2)
        and rho2 <= (lambda1 - lambda2)
        and rho1 >= (lambda1 + lambda2)
    ):
        area_index = 11
        beta1 = rho1 - lambda1 - lambda2
        beta2 = 0
    if rho2 > (lambda1 - lambda2) and rho2 < (rho1 - 2 * lambda2):
        area_index = 12
        beta1 = rho1 - lambda1 - lambda2
        beta2 = rho2 - lambda1 + lambda2

    return [beta1, beta2]
