import numpy as np
from ddn import tools


def prep_sim_from_two_omega(omega1, omega2):
    g1_cov, _ = create_cov_prec_mat(omega1)
    g2_cov, _ = create_cov_prec_mat(omega2)
    comm_gt, diff_gt = tools.get_common_diff_net_topo([omega1, omega2])
    return g1_cov, g2_cov, comm_gt, diff_gt


def gen_sample_two_conditions(g1_cov, g2_cov, n1, n2):
    dat1 = tools.gen_mv(g1_cov, n1)
    dat2 = tools.gen_mv(g2_cov, n2)
    return dat1, dat2


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


def create_chain_graph(N=10, N0=-1, N1=-1, a=1.0):
    # FIXME: use the AR1 covariance structure
    mat_p = np.zeros((N, N))
    if N0 < 0:
        N0 = 0
    if N1 < 0:
        N1 = N - 1
    for i in range(N):
        if i < N0 or i > N1:
            mat_p[i, i] = 1.0
            continue
        if i == 0:
            mat_p[i, i] = a + 1
            continue
        if i > N0:
            j = i - 1
            mat_p[i, j] = -a
            mat_p[j, i] = -a
        if i == N1:
            mat_p[i, i] = a
        else:
            mat_p[i, i] = 2 * a

    m = np.diag(1.0 / np.sqrt(np.diagonal(mat_p)))
    mat_prec = m @ mat_p @ m

    return mat_prec, mat_p


def create_random_graph_two_conditions(N_node, N_edge, N_dif_edge):
    N_edge_union = N_edge + N_dif_edge

    s_max = 10.0
    while s_max >= 1.0:
        edge_union, edge_weight = create_edge_and_weight_union(N_node, N_edge_union)
        g_prec_union, s_max = create_prec_mat_temp(edge_union, edge_weight, N_node)
    print(s_max)

    idx_dif = np.random.choice(N_edge, 2 * N_dif_edge, replace=False)
    idx1 = idx_dif[:N_dif_edge]
    idx2 = idx_dif[-N_dif_edge:]
    g1_prec_org = remove_edges(g_prec_union, edge_union, idx1)
    g2_prec_org = remove_edges(g_prec_union, edge_union, idx2)

    return g1_prec_org, g2_prec_org


def create_edge_and_weight_union(N_node, N_edge_union):
    # edge weights for union of graphs
    x = np.random.uniform(0, 1, N_edge_union)
    if 1:
        x[x < 0.5] = x[x < 0.5] - 1
        edge_weight = x * 0.4
    if 0:
        edge_weight = x * 0
        edge_weight[x >= 0.5] = 0.5
        edge_weight[x < 0.5] = -0.5

    # edges for union of graphs
    edge_all = np.tril_indices(N_node, k=-1)
    edge_union_idx = np.random.choice(len(edge_all[0]), N_edge_union, replace=False)
    edge_union = np.array([edge_all[0][edge_union_idx], edge_all[1][edge_union_idx]])

    return edge_union, edge_weight


def get_graph_subset(edge_union, edge_weight, N_edge, gap=0):
    edge = np.array(
        [edge_union[0][gap : N_edge + gap], edge_union[1][gap : N_edge + gap]]
    )
    edge_wt = edge_weight[gap : N_edge + gap]
    return edge, edge_wt


def remove_edges(g_prec_union, edge_union, idx_lst):
    g_prec_sub = np.copy(g_prec_union)
    for idx in idx_lst:
        i = edge_union[0][idx]
        j = edge_union[1][idx]
        g_prec_sub[i, j] = 0
        g_prec_sub[j, i] = 0
        # print(i, j)
    return g_prec_sub


def create_prec_mat_temp(edge, edge_weight, N_node):
    p = np.zeros((N_node, N_node))
    p[*edge] = edge_weight
    p = p + p.T

    # diagonal dominant
    wt = np.sum(np.abs(p), axis=1)
    wt[np.abs(wt) < 1e-4] = 1.0
    p_diag_dorm = p / wt.reshape((-1, 1)) / 1.5
    p_diag_dorm = (p_diag_dorm + p_diag_dorm.T) / 2
    s = np.sum(np.abs(p_diag_dorm), axis=1)
    s_max = np.max(s)

    p_diag_dorm[np.arange(N_node), np.arange(N_node)] = 1.0

    return p_diag_dorm, s_max


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


def count_edges_from_prec_mat(prec_mat):
    # count the edge number
    prec_mat1 = np.tril(prec_mat, k=-1)
    nz_pairs = np.array(np.where(np.abs(prec_mat1) > 1e-4))
    print(len(nz_pairs[0]))

    return nz_pairs


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
