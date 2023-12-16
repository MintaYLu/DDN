import numpy as np


def standardizeGeneData(genedata, scaler="std", zero_mean=False):
    # sample standardization : z = (x - u) / s

    standarddata = np.zeros(genedata.shape)
    for i in range(genedata.shape[1]):
        # mean value
        u = np.mean(genedata[:, i]) if not zero_mean else 0

        if scaler == "std":
            # standard deviation
            s = np.std(genedata[:, i])
        elif scaler == "rms":
            # root-mean-square
            s = np.sqrt(np.mean(np.square(genedata[:, i])))
        else:
            s = 1

        standarddata[:, i] = (genedata[:, i] - u) / s

    return standarddata


def gen_mv(cov_mat, n_sample):
    x = np.random.multivariate_normal(np.zeros(len(cov_mat)), cov_mat, n_sample)
    return standardizeGeneData(x, scaler="rms")


def concatenateGeneData(controldata, casedata, method="diag"):
    if method == "row":
        return np.concatenate((controldata, casedata), axis=0)
    elif method == "col":
        return np.concatenate((controldata, casedata), axis=1)
    elif method == "diag":
        return np.concatenate(
            (
                np.concatenate((controldata, casedata * 0), axis=0),
                np.concatenate((controldata * 0, casedata), axis=0),
            ),
            axis=1,
        )
    else:
        return []


def get_net_topo_from_mat(mat_prec, thr=1e-4):
    N = len(mat_prec)
    x = np.copy(mat_prec)
    x[np.arange(N), np.arange(N)] = 0.0
    x = 1.0 * (np.abs(x) > thr)
    x = 1.0 * ((x + x.T) > 0)
    return x


def get_common_diff_net_topo(g_beta, thr=1e-4):
    g1 = get_net_topo_from_mat(g_beta[0], thr=thr)
    g2 = get_net_topo_from_mat(g_beta[1], thr=thr)
    g_net_comm = 1.0 * ((g1 + g2) == 2)
    g_net_dif = 1.0 * (g1 != g2)
    return g_net_comm, g_net_dif


def ddn_obj_fun(y, X, lambda1, lambda2, n1, n2, beta):
    p = X.shape[1] // 2
    beta1 = beta[:p]
    beta2 = beta[p:]
    d0 = y - X @ beta
    res = (
        np.sum(d0 * d0) / n1 / 2
        + np.sum(np.abs(beta)) * lambda1
        + np.sum(np.abs(beta1 - beta2)) * lambda2
    )
    return res
