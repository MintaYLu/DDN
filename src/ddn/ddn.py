import numpy as np
import joblib
from joblib import Parallel, delayed
from ddn import tools, solver


def ddn_parallel(
    g1_data,
    g2_data,
    lambda1=0.30,
    lambda2=0.10,
    threshold=1e-6,
    mthd="resi",
    std_est='std',
    g_rec_in=(),
    n_process=1,
):
    if n_process <= 1:
        n_process = int(joblib.cpu_count() / 2)

    n_node = g1_data.shape[1]
    n1 = g1_data.shape[0]
    n2 = g2_data.shape[0]
    g1_data = tools.standardizeGeneData(g1_data, scaler=std_est)
    g2_data = tools.standardizeGeneData(g2_data, scaler=std_est)

    if len(g_rec_in) == 0:
        g_rec_in = np.zeros((2, n_node, n_node))
        use_warm = False
    else:
        use_warm = True

    if mthd == "corr":
        corr_matrix_1 = g1_data.T @ g1_data / n1
        corr_matrix_2 = g2_data.T @ g2_data / n2
    else:
        corr_matrix_1 = []
        corr_matrix_2 = []

    if mthd == "org":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_org)(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    elif mthd == "resi":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_resi)(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    elif mthd == "corr":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_corr)(
                corr_matrix_1,
                corr_matrix_2,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    elif mthd == "strongrule":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_strongrule)(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    else:
        raise ("Method not implemented")

    g_rec = np.zeros((2, n_node, n_node))
    for node in range(n_node):
        g_rec[0, node, :] = out[node][0]
        g_rec[1, node, :] = out[node][1]

    return g_rec


def ddn(
    g1_data,
    g2_data,
    lambda1=0.30,
    lambda2=0.10,
    threshold=1e-6,
    mthd="resi",
    std_est='std',
    g_rec_in=(),
):
    n_node = g1_data.shape[1]
    n1 = g1_data.shape[0]
    n2 = g2_data.shape[0]
    g1_data = tools.standardizeGeneData(g1_data, scaler=std_est)
    g2_data = tools.standardizeGeneData(g2_data, scaler=std_est)

    if len(g_rec_in) == 0:
        g_rec_in = np.zeros((2, n_node, n_node))
        use_warm = False
    else:
        use_warm = True

    if mthd == "corr":
        corr_matrix_1 = g1_data.T @ g1_data / n1
        corr_matrix_2 = g2_data.T @ g2_data / n2
    else:
        corr_matrix_1 = []
        corr_matrix_2 = []

    g_rec = np.zeros((2, n_node, n_node))
    for node in range(n_node):
        beta1_in = g_rec_in[0][node]
        beta2_in = g_rec_in[1][node]

        if mthd == "org":
            beta1, beta2 = solver.run_org(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        elif mthd == "resi":
            beta1, beta2 = solver.run_resi(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        elif mthd == "corr":
            beta1, beta2 = solver.run_corr(
                corr_matrix_1,
                corr_matrix_2,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        elif mthd == "strongrule":
            beta1, beta2 = solver.run_strongrule(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        else:
            print("Method not implemented")
            break

        g_rec[0, node, :] = beta1
        g_rec[1, node, :] = beta2

    return g_rec
