from ddn import ddn, tools_export
from ddn.simulation import simple_data


def ddn_pipeline(dat1, dat2, gene_names, lambda1=0.3, lambda2=0.1):
    omega1, omega2 = ddn.ddn(dat1, dat2, lambda1, lambda2)
    comm_edge, _, _, diff_edge, node_non_isolated = tools_export.get_diff_comm_net_for_plot(omega1, omega2, gene_names)
    return comm_edge, diff_edge, node_non_isolated
