#  import numpy as np

#  from settings import settings as s


#  labels = ("pos_α_vec", "neg_α_vec", "pos_α_ks", "neg_α_ks",
#            "pos_up_bound", "neg_up_bound", "pos_low_bound", "neg_low_bound",
#            "pos_abs_len", "neg_abs_len", "pos_rel_len", "neg_rel_len",
#            "loglr_right", "loglr_left", "loglpv_right", "loglpv_left")


pl_distro_map = {'tpl': "truncated_power_law",
                 'exp': "exponential",
                 'logn': "lognormal"}


def init_tickers_results(tickers):
    """persistent store for computed tail fitting data of all tickers"""
    pass


#  # TODO: use a defaultdict to initialize the data storage container???
# NOTE: len of each list is len(spec_dates) == n_spdt -> use np.ndarray?
def init_results_lists():

    tail_stats_labs = ('α', 'xmin', 'σ', 'abs_len', 'ks_pv')
    loglR_labs = tuple(f'loglR_{distro}' for distro in pl_distro_map)
    loglp_labs = tuple(f'loglp_{distro}' for distro in pl_distro_map)

    # TODO: consider using deque module for fast appending?
    return {lab: [] for lab in tail_stats_labs + loglR_labs + loglp_labs}


def boxplot_mat_init():
    labels = ("pos_α_mat", "neg_α_mat")
    return {l: [] for l in labels}


def gen_vector_colnames(tails_used):
    pass
