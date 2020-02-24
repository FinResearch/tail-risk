import numpy as np

from powerlaw import Fit
from ._plpva import plpva as _plpva

# TODO: rid dependency on settings module? --> use OOP
from .settings import settings as s


#  pl_distro_map = {'tpl': "truncated_power_law",
#                   'exp': "exponential",
#                   'logn': "lognormal"}

distros = ('truncated_power_law', 'exponential', 'lognormal')


# ## taken from utils.py

def __get_xmin(data, xmin=None):

    # NOTE: if explicitly passed xmin to function, then ofc use it
    if xmin is not None:  # NOTE: use this for Rolling
        return xmin

    # TODO: don't use global s settings var
    xmin_rule, xmin_val = s.xmin_inputs

    if xmin_rule == "clauset":
        xmin = None
    elif xmin_rule == "manual":
        xmin = xmin_val
    elif xmin_rule == "percentile":
        xmin = np.percentile(data, xmin_val)

    return xmin


def _get_Fit(data, xmin=None):

    xmin = __get_xmin(data, xmin)

    # NOTE: only keep/use non-zero elements
    data = np.array(data)[np.nonzero(data)]

    # data_nature is one of ['discrete', 'continuous']
    discrete = False if s.data_nature == 'continuous' else True

    return Fit(data, discrete=discrete, xmin=xmin)


#  def get_fit(data):
#
#      fit = init_fit(data)
#
#      # TODO: test standardization branch
#      if s.standardize and s.standardize_target == "tail":
#          xmin = fit.power_law.xmin
#          fit = standardize_tail(data, xmin)
#
#      return fit


# FIXME: need to have good way to standardize & absolutize tails
#  # TODO: test standardization function
#  def standardize_tail(data):
#      print("I am standardizing your data")
#      S = np.array(data)
#      m = np.mean(S)
#      v = np.std(S)
#      X = (S - m) / v
#      return X
#
#  # TODO: test absolutize function
#  def absolutize_tail(data):
#      print("I am taking the absolute value of your data")
#      return np.abs(data)


# configure given series to chosen return_type
def _config_series(series):

    # TODO: add fullname for return_types, ex. {"log": "Log Returns"}
    print(f"You opted for the analysis of the {s.return_type}")

    tau = s.tau

    pt_f = series[tau:]
    pt_i = series[0: len(series) - tau]

    if s.return_type == "basic":
        X = pt_f - pt_i
    elif s.return_type == "relative":
        X = pt_f / pt_i - 1.0
    elif s.return_type == "log":
        X = np.log(pt_f/pt_i)

    if s.standardize is True:
        print("I am standardizing your time series")
        X = (X - X.mean())/X.std()

    if s.absolutize is True:
        print("I am taking the absolute value of your time series")
        X = X.abs()

    return X


def _get_tail_stats(fit, series):

    fpl = fit.power_law  # fpl: fitted power law
    alpha = fpl.alpha
    xmin = fpl.xmin
    sigma = fpl.sigma

    abs_len = len(series[series >= xmin])
    ks_pv, _ = _plpva(series, xmin, 'reps', s.plpva_iter, 'silent')

    return [alpha, xmin, sigma, abs_len, ks_pv]


def _get_logl_tstats(fit):

    logl_stats = []
    for pdf in distros:
        R, p = fit.distribution_compare("power_law", pdf,
                                        normalized_ratio=True)
        logl_stats.append(R)
        logl_stats.append(p)

    return logl_stats


def get_results_tup(series):

    X = _config_series(series)

    trl = []  # trl: tails results list
    for tdir in s.tails_used:
        data = X if tdir == 'right' else -X
        fit = _get_Fit(data)

        tail_stats = _get_tail_stats(fit, data)
        logl_stats = _get_logl_tstats(fit)

        trl.append(tail_stats + logl_stats)

    return [stat for stat_tup in zip(*trl) for stat in stat_tup]


def get_extra_plot_vecs(results_mat):
    pass
    #  extras = ('up_bound', 'low_bound', 'rel_len')
    #  bound_delta = s.alpha_quantile * s_err
    #  up_bound = alpha + bound_delta
    #  low_bound = alpha - bound_delta

    #  abs_len = len(tail[tail >= xmin])
    #  # TODO: rel_len is just abs_len / const known in advance
    #  rel_len = len(tail[tail >= xmin]) / len(tail)
