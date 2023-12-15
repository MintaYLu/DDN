import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist
import networkx as nx
import matplotlib.pyplot as plt


def draw_network_for_ddn(
    edges_df,
    nodes_to_draw,
    fig_size=12,
    font_size_scale=1,
    node_size_scale=2,
    part_number=1,
    nodes_type=None,
    labels=None,
    mode="common",
    export_pdf=True,
    pdf_name="",
):
    # create networkx graph
    G = _create_nx_graph(
        edges_df,
        nodes_to_draw,
        mode=mode,
        nodes_type=nodes_type,
    )

    # nodes positions
    # must provide nodes_type dictionary for two parts
    if part_number == 1:
        pos, d_min = _get_pos_one_part(nodes_to_draw)
    elif part_number == 2:
        pos, d_min = _get_pos_two_parts(nodes_to_draw, nodes_type)
    else:
        raise ("Not implemented")

    # plot the network
    _plot_network_helper(
        G,
        pos,
        d_min=d_min,
        labels=labels,
        fig_size=fig_size,
        font_size_scale=font_size_scale,
        node_size_scale=node_size_scale,
    )

    # export figure
    if export_pdf:
        plt.savefig(f"{pdf_name}_{mode}.pdf", format="pdf", bbox_inches="tight")

    return G


def _add_node_to_a_circle(pos, nodes, cen, rad, angles):
    pos_lst = []
    for i, node in enumerate(nodes):
        theta = angles[i]
        pos0 = np.array(
            [cen[0] + np.cos(theta) * rad[0], cen[1] + np.sin(theta) * rad[1]]
        )
        pos[node] = pos0
        pos_lst.append(pos0)
    pos_lst = np.array(pos_lst)
    if len(pos_lst) > 1:
        d = pdist(pos_lst)
        return np.min(d)
    else:
        return 0.5


def _angles_in_ellipse(num, a, b):
    # Based on https://stackoverflow.com/a/52062369, 
    # which is from https://pypi.org/project/flyingcircus/
    assert num > 0
    assert a < b
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e2 = 1.0 - a**2.0 / b**2.0
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e2)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(lambda x: (sp.special.ellipeinc(x, e2) - arcs), angles)
        angles = res.x
    return angles


def _get_pos_one_part(nodes_show):
    # positions of nodes
    n = len(nodes_show)

    # generate positions
    angle = _angles_in_ellipse(n, 0.999, 1.0)
    cen = [0.0, 0]
    rad = [1.0, 1.0]

    pos = dict()
    d_min = _add_node_to_a_circle(pos, nodes_show, cen, rad, angle)
    return pos, d_min


def _get_pos_two_parts(nodes_show, nodes_type):
    # positions of nodes
    nodes_sp = []
    nodes_tf = []
    for node in nodes_show:
        if nodes_type[node] == 0:
            nodes_sp.append(node)
        if nodes_type[node] == 1:
            nodes_tf.append(node)

    n_sp = len(nodes_sp)
    n_tf = len(nodes_tf)

    # generate positions
    angle_sp = _angles_in_ellipse(n_sp, 0.4, 1.0)
    angle_tf = _angles_in_ellipse(n_tf, 0.4, 1.0)

    cen_sp = [-0.6, 0]
    cen_tf = [0.6, 0]
    rad_sp = [0.4, 1]
    rad_tf = [0.4, 1]

    pos = dict()
    d_min_sp = _add_node_to_a_circle(pos, nodes_sp, cen_sp, rad_sp, angle_sp)
    d_min_tf = _add_node_to_a_circle(pos, nodes_tf, cen_tf, rad_tf, angle_tf)
    d_min = min(d_min_sp, d_min_tf)
    return pos, d_min


def _create_nx_graph(
    edges_df,
    nodes_show,
    min_alpha=0.2,
    max_alpha=1.0,
    mode="common",
    nodes_type=None,
):
    # create the overall graph
    color_condition = {0: [0.7, 0.7, 0.7], 1: [0, 0, 1], 2: [1, 0, 0], 3: [0, 0.6, 0.3]}
    beta_max = np.max(edges_df["weight"])

    if nodes_type is None:
        nodes_type = dict()
        for node in nodes_show:
            nodes_type[node] = 0

    G = nx.Graph()
    for node in nodes_show:
        G.add_node(node)

    for i in range(len(edges_df)):
        gene1, gene2, condition, beta = edges_df.iloc[i]
        if condition in color_condition:
            alpha = np.abs(beta) / beta_max * (max_alpha - min_alpha) + min_alpha
            weight = np.abs(beta) / beta_max * 3.0 + 0.5
            if mode != "common":
                color = list(1 - (1 - np.array(color_condition[condition])) * alpha)
            else:
                if nodes_type[gene1] == nodes_type[gene2]:
                    color = list(1 - (1 - np.array(color_condition[0])) * alpha)
                else:
                    color = list(1 - (1 - np.array(color_condition[3])) * alpha)
            G.add_edge(gene1, gene2, color=color, weight=weight)

    return G


def _plot_network_helper(
    G,
    pos,
    d_min,
    labels,
    fig_size=18,
    font_size_scale=1,
    node_size_scale=2,
):
    s_min = (d_min * 100) ** 2
    # print("s_min ", s_min)
    s_min = min(s_min, 500)
    node_size = np.array([d for n, d in G.degree()])
    node_size = node_size / (np.max(node_size)+1)

    node_size = node_size * s_min * node_size_scale * 10
    font_size = d_min * font_size_scale * 100
    font_size = min(font_size, 10)
    # font_size = s_min * font_size_scale * 0.5

    # edges properties
    edges = G.edges()
    edge_color = [G[u][v]["color"] for u, v in edges]
    edge_weight = np.array([G[u][v]["weight"] for u, v in edges])
    if len(edge_weight) > 400:
        edge_weight = edge_weight / len(edge_weight) * 400

    # draw
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.5,
    )

    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        edgelist=edges,
        edge_color=edge_color,
        width=edge_weight,
    )

    nx.draw_networkx_labels(
        G,
        pos=pos,
        ax=ax,
        labels=labels,
        font_size=font_size,
        font_color="blueviolet",
    )

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim(ax.get_xlim())
