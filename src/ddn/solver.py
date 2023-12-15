import numpy as np
from ddn import tools
from ddn import bcd, strong_rule


def run_org(
    g1_data,
    g2_data,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold=1e-6,
    use_warm=False,
):
    beta_in = np.concatenate(
        (
            beta1_in[:node],
            beta1_in[node + 1 :],
            beta2_in[:node],
            beta2_in[node + 1 :],
        )
    )

    N_NODE = g1_data.shape[1]
    n1, n2 = g1_data.shape[0], g2_data.shape[0]

    y = tools.concatenateGeneData(g1_data[:, node], g2_data[:, node], method="row")
    node_fea = [i for i in range(N_NODE) if i != node]
    X = tools.concatenateGeneData(
        g1_data[:, node_fea], g2_data[:, node_fea], method="diag"
    )
    beta, r, betaerr = bcd.bcd_org(beta_in, y, X, lambda1, lambda2, n1, n2, threshold)
    # beta, r, betaerr = bcd.bcd_org_old(beta_in, y, X, lambda1, lambda2, n1, n2, threshold)

    # print(r, betaerr)

    # reindex the features
    beta1 = list(beta[0:node]) + [0] + list(beta[node : N_NODE - 1])
    beta1 = np.array(beta1)
    beta2 = (
        list(beta[N_NODE - 1 : node + N_NODE - 1])
        + [0]
        + list(beta[node + N_NODE - 1 : 2 * N_NODE - 2])
    )
    beta2 = np.array(beta2)
    return beta1, beta2


def run_resi(
    g1_data,
    g2_data,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
    use_warm=False,
):
    beta_in = np.concatenate((beta1_in, beta2_in))
    if use_warm:
        beta1_in_x = np.copy(beta1_in)
        beta2_in_x = np.copy(beta2_in)
        beta1_in_x[node] = 0
        beta2_in_x[node] = 0
        y1_resi = g1_data[:, node] - np.dot(g1_data, beta1_in_x)
        y2_resi = g2_data[:, node] - np.dot(g2_data, beta2_in_x)
    else:
        y1_resi = g1_data[:, node]
        y2_resi = g2_data[:, node]

    N_NODE = g1_data.shape[1]

    # beta = bcd_residual(g1_data, g2_data, node, lambda1, lambda2, threshold)
    beta, r, betaerr = bcd.bcd_residual(
        beta_in,
        g1_data,
        g2_data,
        y1_resi,
        y2_resi,
        node,
        lambda1,
        lambda2,
        threshold,
    )
    # n_iter.append(r)
    beta1 = np.array(beta[:N_NODE])
    beta2 = np.array(beta[N_NODE:])

    if 0:
        print("Convergence: ", r, betaerr)

    return beta1, beta2


def run_corr(
    corr_matrix_1,
    corr_matrix_2,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
    use_warm=False,
):
    beta_in = np.concatenate((beta1_in, beta2_in))

    beta, r, betaerr = bcd.bcd_corr(
        beta_in,
        node,
        lambda1,
        lambda2,
        corr_matrix_1,
        corr_matrix_2,
        threshold=threshold,
        max_iter=100000,
    )

    n_node = len(beta1_in)
    beta1 = np.array(beta[:n_node])
    beta2 = np.array(beta[n_node:])

    return beta1, beta2


def run_strongrule(
    g1_data,
    g2_data,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
    use_warm,
):
    beta1_in_x = (np.concatenate([beta1_in[:node], beta1_in[node + 1 :]]),)
    beta2_in_x = (np.concatenate([beta2_in[:node], beta2_in[node + 1 :]]),)
    y1 = g1_data[:, node]
    y2 = g2_data[:, node]

    N_NODE = g1_data.shape[1]

    # choose other genes as feature
    idx = [i for i in range(N_NODE) if i != node]
    X1 = g1_data[:, idx]
    X2 = g2_data[:, idx]

    # perform bcd algorithm
    beta1 = strong_rule.lasso(
        y1,
        X1,
        lambda1,
        beta1_in_x,
        tol=threshold,
        use_strong_rule=True,
        use_warm=use_warm,
    )
    beta2 = strong_rule.lasso(
        y2,
        X2,
        lambda1,
        beta2_in_x,
        tol=threshold,
        use_strong_rule=True,
        use_warm=use_warm,
    )

    # reindex the features
    beta1 = list(beta1[0:node]) + [0] + list(beta1[node:])
    beta1 = np.array(beta1)
    beta2 = list(beta2[0:node]) + [0] + list(beta2[node:])
    beta2 = np.array(beta2)

    return beta1, beta2

