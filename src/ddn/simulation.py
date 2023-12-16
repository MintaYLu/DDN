"""Utility function for simple simulation

These simulations are designed to illustrate the usage DDN, and not for formal evaluation of performance.
Only the pair graph is supported.

It also contains some helper functions for other simulation functions.

"""

import numpy as np
from ddn import tools


def create_pair_graph(
    n_node=40,
    corr=0.75,
    n_shuf=3,
):
    n_blk = int(n_node / 2)
    edge1 = np.zeros((n_blk, 2))
    edge1[:, 0] = np.arange(n_blk)
    edge1[:, 1] = np.arange(n_blk) + n_blk

    edge2 = np.copy(edge1)
    idx_shuf = np.arange(n_blk, n_blk + n_shuf)
    edge2[: len(idx_shuf), 1] = np.concatenate([idx_shuf[[-1]], idx_shuf[:-1]])

    xx = np.array([[1, -corr], [-corr, 1]])
    # print(np.linalg.eig(xx))

    g1_prec = np.zeros((n_node, n_node))
    for e in edge1:
        e = e.astype(int)
        g1_prec[np.ix_([e[0], e[1]], [e[0], e[1]])] = xx

    g2_prec = np.zeros((n_node, n_node))
    for e in edge2:
        e = e.astype(int)
        g2_prec[np.ix_([e[0], e[1]], [e[0], e[1]])] = xx

    return g1_prec, g2_prec


def prep_sim_from_two_omega(omega1, omega2):
    g1_cov, _ = create_cov_prec_mat(omega1)
    g2_cov, _ = create_cov_prec_mat(omega2)
    comm_gt, diff_gt = tools.get_common_diff_net_topo([omega1, omega2])
    return g1_cov, g2_cov, comm_gt, diff_gt


def gen_sample_two_conditions(g1_cov, g2_cov, n1, n2):
    dat1 = tools.gen_mv(g1_cov, n1)
    dat2 = tools.gen_mv(g2_cov, n2)
    return dat1, dat2


def create_cov_prec_mat(prec_mat_temp):
    """Create covariance and precision matrix from temporary precision matrix

    We follow [Peng 2009] and do not use the d_ij term as the JGL paper.

    Args:
        prec_mat_temp (_type_): _description_

    Returns:
        _type_: _description_
    """
    cov_mat_temp = np.linalg.inv(prec_mat_temp)
    d_sqrt = np.sqrt(np.diag(1 / np.diagonal(cov_mat_temp)))
    cov_mat = d_sqrt @ cov_mat_temp @ d_sqrt
    prec_mat = np.linalg.inv(cov_mat)

    return cov_mat, prec_mat


def simple_data():
    n_node = 40
    n_sample1 = 100
    n_sample2 = 100
    n_shuf = 5
    g1_prec, g2_prec = create_pair_graph(n_node=n_node, corr=0.75, n_shuf=n_shuf)
    gene_names = [f"Gene{i}" for i in range(n_node)]
    g1_cov, _ = create_cov_prec_mat(g1_prec)
    g2_cov, _ = create_cov_prec_mat(g2_prec)
    dat1 = tools.gen_mv(g1_cov, n_sample1)
    dat2 = tools.gen_mv(g2_cov, n_sample2)

    return dat1, dat2, gene_names
