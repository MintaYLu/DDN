import numpy as np
import pandas as pd
from ddn import tools


def get_diff_comm_net_for_plot(omega1, omega2, gene_names):
    # Get edges in format `gene1`, `gene2`, `condition`, `beta`
    omega1 = (omega1 + omega1.T) / 2
    omega2 = (omega2 + omega2.T) / 2
    omega_comm = (omega1 + omega2) / 2

    g1 = tools.get_net_topo_from_mat(omega1)
    g2 = tools.get_net_topo_from_mat(omega2)
    gene_names = np.array(gene_names)

    comm_adj_mat = (g1 + g2) == 2
    dif1_adj_mat = g1 != comm_adj_mat
    dif2_adj_mat = g2 != comm_adj_mat

    comm_edge = _get_edge_list(comm_adj_mat, omega_comm, gene_names, group_idx=0)
    dif1_edge = _get_edge_list(dif1_adj_mat, omega1, gene_names, group_idx=1)
    dif2_edge = _get_edge_list(dif2_adj_mat, omega2, gene_names, group_idx=2)

    node_degree_nz = (np.sum(g1, axis=0) > 0) + (np.sum(g2, axis=0) > 0)
    node_idx_non_isolated = np.sort(np.where(node_degree_nz > 0)[0])
    node_non_isolated = gene_names[node_idx_non_isolated]
    diff_edge = pd.concat((dif1_edge, dif2_edge))

    return comm_edge, dif1_edge, dif2_edge, diff_edge, node_non_isolated


def _get_edge_list(conn_mat, beta_mat, gene_names, group_idx=0):
    conn_mat1 = np.tril(conn_mat, -1)
    n1, n2 = np.where(conn_mat1 > 0)
    edge_weight = np.abs(beta_mat[n1, n2])
    out = dict(
        gene1=gene_names[n1],
        gene2=gene_names[n2],
        condition=group_idx,
        weight=edge_weight,
    )
    return pd.DataFrame(data=out)


def get_node_type_and_label_two_parts(
    nodes_show, part1_id="SP", part2_id="TF", ofst1=2, ofst2=1
):
    # TODO: support more parts
    x_len1 = len(part1_id)
    x_len2 = len(part2_id)

    # make labels shorter
    nodes_type = dict()
    labels = dict()
    for i, node in enumerate(nodes_show):
        if node[:x_len1] == part1_id:
            labels[node] = node[x_len1 + ofst1 :]
            nodes_type[node] = 0
        if node[:x_len2] == part2_id:
            labels[node] = node[x_len2 + ofst2 :]
            nodes_type[node] = 1

    return nodes_type, labels


def get_edge_subset_by_name(edge_df, name_match="TF"):
    gene_sel = np.zeros(len(edge_df))
    nn = len(name_match)
    for i in range(len(edge_df)):
        gene1, gene2, _, _ = edge_df.iloc[i]
        if gene1[:nn] == name_match and gene2[:nn] == name_match:
            gene_sel[i] = 1
    return edge_df[gene_sel > 0]


def get_node_subset_by_name(nodes, name_match="TF"):
    nodes_sel = []
    nn = len(name_match)
    for node in nodes:
        if node[:nn] == name_match:
            nodes_sel.append(node)
    return nodes_sel
