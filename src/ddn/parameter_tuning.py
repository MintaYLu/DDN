import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ddn import ddn
from ddn import tools 


class DDNParameterSearch:
    def __init__(
        self,
        dat1,
        dat2,
        lambda1_list=np.arange(0.05, 1.05, 0.05),
        lambda2_list=np.arange(0.025, 0.525, 0.025),
        n_cv=5,
        ratio_validation=0.2,
        alpha1=0.05,
        alpha2=0.01,
    ) -> None:
        # parameters
        self.dat1 = dat1
        self.dat2 = dat2
        self.l1_lst = lambda1_list
        self.l2_lst = lambda2_list
        self.n_cv = n_cv
        self.ratio_val = ratio_validation
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # derived and outputs
        self.n = int((dat1.shape[0] + dat2.shape[0]) / 2)
        self.p = dat1.shape[1]

    def fit(self, method='cv_sequential'):
        if method == 'cv_joint':
            out = self.run_cv_joint()
        elif method == 'cv_sequential':
            out = self.run_cv_sequential()
        elif method == 'cv_bai':
            out = self.run_cv_bai()
        elif method == 'mb_cv':
            out = self.run_mb_cv()
        elif method == 'mb_bai':
            out = self.run_mb_bai()
        else:
            out = self.run_cv_bai()
        return out

    def run_cv_joint(self):
        # can be slow
        val_err, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
            lambda1_lst=self.l1_lst,
            lambda2_lst=self.l2_lst,
        )
        l1_est, l2_est = get_lambdas_one_se_2d(val_err, self.l1_lst, self.l2_lst)
        return val_err, l1_est, l2_est

    def run_cv_sequential(self):
        val_err1, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=self.l1_lst,
            lambda2_lst=[0.0],
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err1 = np.squeeze(val_err1)
        l1_est = get_lambda_one_se_1d(val_err1, self.l1_lst)

        val_err2, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=[l1_est],
            lambda2_lst=self.l2_lst,
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err2 = np.squeeze(val_err2)
        l2_est = get_lambda_one_se_1d(val_err2, self.l2_lst)

        return [val_err1, val_err2], l1_est, l2_est

    def run_cv_bai(self):
        val_err, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=self.l1_lst,
            lambda2_lst=[0.0],
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err = np.squeeze(val_err)
        l1_est = get_lambda_one_se_1d(val_err, self.l1_lst)
        l2_est = get_lambda2_bai(self.dat1, self.dat2, alpha=self.alpha2)

        return val_err, l1_est, l2_est

    def run_mb_cv(self):
        # mb not reliable with small sample size
        l1_est = get_lambda1_mb(self.alpha1, self.n, self.p)
        val_err, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=[l1_est],
            lambda2_lst=self.l2_lst,
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err = np.squeeze(val_err)
        l2_est = get_lambda_one_se_1d(val_err, self.l2_lst)

        return val_err, l1_est, l2_est

    def run_mb_bai(self):
        # mb not reliable with small sample size
        l1_est = get_lambda1_mb(self.alpha1, self.n, self.p, mthd=0)
        l2_est = get_lambda2_bai(self.dat1, self.dat2, alpha=self.alpha2)
        return [], l1_est, l2_est


def plot_error_1d(val_err, lambda_lst=[], ymin=None, ymax=None):
    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]
    if not ymin:
        ymin = max(np.min(val_err_mean), 0.0)
    if not ymax:
        ymax = min(np.max(val_err_mean), 1.0)
    gap1 = (ymax - ymin)/10
    fig, ax = plt.subplots()
    ax.errorbar(
        lambda_lst,
        val_err_mean,
        yerr=val_err_std / np.sqrt(n_cv),
        ecolor="red",
        elinewidth=0.5,
    )
    ax.set_ylim([ymin-gap1, ymax+gap1])


def plot_error_2d(val_err, cmin=None, cmax=None):
    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]
    if not cmin:
        cmin = np.min(val_err_mean)
    if not cmax:
        cmax = np.max(val_err_mean)
    fig, ax = plt.subplots()
    pos = ax.imshow(val_err_mean, origin="lower", vmin=cmin, vmax=cmax)
    color_bar = fig.colorbar(pos, ax=ax)
    ax.set_xlabel('$\lambda_2$')
    ax.set_ylabel('$\lambda_1$')


def get_lambda1_mb(alpha, n, p, mthd=0):
    lmb1 = 0.5
    if mthd == 0:
        # The MB paper do not have 1/2 factor for data term
        # To make things consistent, we further divide by 2
        lmb1 = 2 / np.sqrt(n) * norm.ppf(1 - alpha / (2 * p * n * n)) / 2
        # lmb1 = 2 / np.sqrt(n) * norm.ppf(1 - alpha / (2 * p * n * n))
    if mthd == 1:
        # NOTE: this is from KDDN Java code, which is not the same as MB paper
        lmb1 = 2 / n * norm.ppf(1 - alpha / (2 * p * n * n))
    return lmb1


def get_lambda2_bai(
    x1,
    x2,
    alpha=0.01,
):
    x1 = tools.standardizeGeneData(x1)
    x2 = tools.standardizeGeneData(x2)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    N = (n1 + n2) / 2
    p = x1.shape[1]

    s = norm.ppf(1 - alpha / 2) * np.sqrt(2 / (N - 3))

    # pd = (x1.T @ x1) * (x2.T @ x2)
    pd = (x1.T @ x1 / n1) * (x2.T @ x2 / n2)
    pd[np.arange(p), np.arange(p)] = 0
    rho1rho2 = np.sum(pd) / p / (p - 1)

    lmb2 = (np.exp(2 * s) - 1) / (np.exp(2 * s) + 1) / 2 * (1 - rho1rho2)
    print("Avg rho is ", rho1rho2)
    print("lambda2 is ", lmb2)

    return lmb2


def get_lambda_one_se_1d(val_err, lambda_lst):
    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]

    idx = np.argmin(val_err_mean)
    val_err_mean[:idx] = -10000

    cut_thr = val_err_mean[idx] + val_err_std[idx] / np.sqrt(n_cv)  # standard error
    if np.max(val_err_mean) < cut_thr:
        print("One SE rule failed.")
        return lambda_lst[-1]
    else:
        # idx_thr = np.max(np.where(val_err_mean <= cut_thr)[0])
        idx_thr = np.min(np.where(val_err_mean >= cut_thr)[0])
        return lambda_lst[idx_thr]


def get_lambdas_one_se_2d(val_err, lambda1_lst, lambda2_lst):
    gap1 = np.abs(lambda1_lst[-1] - lambda1_lst[0]) / len(lambda1_lst)
    gap2 = np.abs(lambda2_lst[-1] - lambda2_lst[0]) / len(lambda2_lst)
    lmb1_scale = gap1 / gap2

    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]

    idx1, idx2 = np.unravel_index(val_err_mean.argmin(), val_err_mean.shape)
    print(idx1, idx2)
    print("l1_org, l2_org ", lambda1_lst[idx1], lambda2_lst[idx2])
    msk = np.zeros_like(val_err_mean)
    msk[idx1:, idx2:] = 1.0
    se = val_err_std[idx1, idx2] / np.sqrt(n_cv)
    z = (val_err_mean - np.min(val_err_mean)) / se
    z = z * msk

    if np.max(z) == 0:
        print("One SE rule failed.")
        return np.max(lambda1_lst), np.max(lambda2_lst)
    else:
        m1, m2 = val_err_mean.shape
        cord1, cord2 = np.meshgrid(np.arange(m1), np.arange(m2), indexing="ij")
        d = (cord1 - idx1) ** 2 * lmb1_scale + (cord2 - idx2) ** 2
        d[z < 1] = 100000
        idx1a, idx2a = np.unravel_index(d.argmin(), d.shape)
        print(idx1a, idx2a)
        print("l1, l2 ", lambda1_lst[idx1a], lambda2_lst[idx2a])

        return lambda1_lst[idx1a], lambda2_lst[idx2a]


def calculate_regression(data, topo_est):
    """Linear regression

    # x = np.array([[-1,-1,1,1.0], [1,1,-1,-1]]).T
    # y = np.array([1,1,-1,-1.0])
    # out = np.linalg.lstsq(x, y, rcond=None)
    # out[0]

    Args:
        data (_type_): _description_
        topo_est (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_fea = data.shape[1]
    g_asso = np.eye(n_fea, dtype=np.double)
    for i in range(n_fea):
        pred_idx = np.where(topo_est[i] > 0)[0]
        if len(pred_idx) == 0:
            continue
        y = data[:, i]
        x = data[:, pred_idx]
        out = np.linalg.lstsq(x, y, rcond=None)
        g_asso[i, pred_idx] = out[0]
    return g_asso


def cv_two_lambda(
    dat1,
    dat2,
    n_cv=5,
    ratio_val=0.2,
    lambda1_lst=np.arange(0.05, 1.05, 0.05),
    lambda2_lst=np.arange(0.025, 0.525, 0.025),
):
    val_err = np.zeros((n_cv, len(lambda1_lst), len(lambda2_lst)))
    n_node = dat1.shape[1]
    n1 = dat1.shape[0]
    n2 = dat2.shape[0]
    n1_val = int(n1 * ratio_val)
    n1_train = n1 - n1_val
    n2_val = int(n2 * ratio_val)
    n2_train = n2 - n2_val

    mthd = 'resi'
    # if len(lambda2_lst) == 1 and lambda2_lst[0] == 0:
    #     mthd = 'strongrule' 
    print(mthd)       

    for n in range(n_cv):
        print("Repeat ============", n)
        msk1 = np.zeros(n1)
        msk1[np.random.choice(n1, n1_train, replace=False)] = 1
        msk2 = np.zeros(n2)
        msk2[np.random.choice(n2, n2_train, replace=False)] = 1
        g1_train = tools.standardizeGeneData(dat1[msk1 > 0])
        g1_val = tools.standardizeGeneData(dat1[msk1 == 0])
        g2_train = tools.standardizeGeneData(dat2[msk2 > 0])
        g2_val = tools.standardizeGeneData(dat2[msk2 == 0])

        for i, lambda1 in enumerate(lambda1_lst):
            # print(n, i)
            for j, lambda2 in enumerate(lambda2_lst):
                g_beta_est = ddn.ddn(
                    g1_train,
                    g2_train,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    mthd=mthd,
                )
                g1_net_est = tools.get_net_topo_from_mat(g_beta_est[0])
                g2_net_est = tools.get_net_topo_from_mat(g_beta_est[1])
                g1_coef = calculate_regression(g1_train, g1_net_est)
                g1_coef[np.arange(n_node), np.arange(n_node)] = 0
                g2_coef = calculate_regression(g2_train, g2_net_est)
                g2_coef[np.arange(n_node), np.arange(n_node)] = 0
                rec_ratio1 = np.linalg.norm(
                    g1_val @ g1_coef.T - g1_val
                ) / np.linalg.norm(g1_val)
                rec_ratio2 = np.linalg.norm(
                    g2_val @ g2_coef.T - g2_val
                ) / np.linalg.norm(g2_val)
                val_err[n, i, j] = (rec_ratio1 + rec_ratio2) / 2

    return val_err, lambda1_lst, lambda2_lst
