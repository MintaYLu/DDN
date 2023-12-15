import numpy as np
from ddn import tools


def scan_error_measure(t1_lst, t2_lst, comm_gt, diff_gt):
    res_comm = np.zeros((len(t1_lst), 5))
    res_diff = np.zeros((len(t1_lst), 5))
    for i in range(len(t1_lst)):
        comm_est, diff_est = tools.get_common_diff_net_topo([t1_lst[i], t2_lst[i]])
        res_comm[i] = get_error_measure_two_theta(comm_est, comm_gt)
        res_diff[i] = get_error_measure_two_theta(diff_est, diff_gt)
    return res_comm, res_diff


def scan_erro_measure_dingo(comm_dingo, diff_dingo, comm_gt, diff_gt):
    thr_rg_comm = np.arange(0, 1, 0.01)
    res_comm_dingo = np.zeros((len(thr_rg_comm), 5))
    for i, thr in enumerate(thr_rg_comm):
        comm_est = tools.get_net_topo_from_mat(comm_dingo, thr=thr)
        res_comm_dingo[i] = get_error_measure_two_theta(comm_est, comm_gt)

    thr_rg_diff = np.arange(0, 50, 0.1)
    res_diff_dingo = np.zeros((len(thr_rg_diff), 5))
    for i, thr in enumerate(thr_rg_diff):
        diff_est = tools.get_net_topo_from_mat(diff_dingo, thr=thr)
        res_diff_dingo[i] = get_error_measure_two_theta(diff_est, diff_gt)
    return res_comm_dingo, res_diff_dingo


def get_error_measure_two_theta(net_est, net_gt):
    # See, e.g., https://en.wikipedia.org/wiki/Confusion_matrix for details
    n_node = len(net_est)
    n_edge = n_node * (n_node-1) / 2

    P = np.sum(net_gt) / 2
    N = n_edge - P

    # number of true positive edges (TP) and false positive edges (FP)
    TP = np.sum((net_est > 0) * (net_gt > 0)) / 2
    FP = np.sum((net_est > 0) * (net_gt == 0)) / 2
    TP_FP = TP + FP

    # true positive rate, recall
    if P > 0:
        TPR = TP / P
    else:
        TPR = 0

    # false positive rate
    if N > 0:
        FPR = FP / N
    else:
        FPR = 0

    # precision
    if TP_FP > 0:
        PPV = TP / TP_FP
    else:
        PPV = 0    

    return np.array([TP, FP, TPR, FPR, PPV])


def get_f1(recall, precision):
    recall1 = np.copy(recall)
    recall1[recall==0] = 1e-8
    f1 = 2*recall1*precision/(recall1+precision)
    f1[recall==0] = 0
    f1[precision==0] = 0
    return f1


def get_f1_mat(a_in):
    n_l2 = len(a_in)
    n_l1 = a_in.shape[1]
    f1_mat = np.zeros((n_l2, 2, n_l1))
    for i in range(n_l2):
        f1_comm = get_f1(a_in[i][:,0,2], a_in[i][:,0,4])
        f1_diff = get_f1(a_in[i][:,1,2], a_in[i][:,1,4])
        f1_mat[i, 0] = f1_comm
        f1_mat[i, 1] = f1_diff
    return f1_mat
