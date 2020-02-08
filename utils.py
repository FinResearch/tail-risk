import numpy as np
from settings import settings as s

import powerlaw as pl


# Common functions in both if-branches

def _get_xmin(data, xmin=None):

    if xmin is not None:  # NOTE: use this for Rolling
        return xmin

    if s.xmin_rule == "Clauset":
        xmin = None
    elif s.xmin_rule == "Manual":
        xmin = s.xmin_value
    elif s.xmin_rule == "Percentile":
        xmin = np.percentile(data, s.xmin_sign)

    return xmin


def powerlaw_fit(data, xmin=None):

    # NOTE: only keep/use non-zero elements
    data1 = np.array(data)[np.nonzero(data)]

    # data_nature is one of ['Discrete', 'Continuous']
    discrete = False if s.data_nature == 'Continuous' else True

    xmin = _get_xmin(data, xmin)

    #  if s.xmin_rule == "Clauset":
    #      xmin = None
    #  elif s.xmin_rule == "Manual":
    #      xmin = s.xmin_value
    #  elif s.xmin_rule == "Percentile":
    #      xmin = np.percentile(data, s.xmin_sign)

    return pl.Fit(data1, discrete=discrete, xmin=xmin)


def fit_tail(data):

    fit = powerlaw_fit(data)

    # TODO: test standardization branch
    if s.standardize and s.standardize_target == "Tail":
        xmin = fit.power_law.xmin
        fit = standardize_tail(data, xmin)

    return data, fit


# TODO: test standardization function
def standardize_tail(tail_data, xmin):
    print("I am standardizing your tail")
    S = np.array(tail_data)
    S = S[S >= xmin]
    m = np.mean(S)
    v = np.std(S)
    X = (S - m) / v

    if s.absolutize and s.absz_target == "Tail":
        X = absolutize_tail(X)

    return powerlaw_fit(X, s.xmin_rule, np.min(X))


# TODO: test absolutize function
def absolutize_tail(tail_data):
    print("I am taking the absolute value of your tail")
    #  lab = "|" + lab + "|"
    return np.abs(tail_data)


#  def get_tail_stats(fit_obj, tail_data, ks_pvgof_tup):
#      alpha = fit_obj.power_law.alpha
#      xmin = fit_obj.power_law.xmin
#      s_err = fit_obj.power_law.sigma
#      tail_size = len(tail_data[tail_data >= xmin])
#      ks_pv = ks_pvgof_tup[0]
#      return alpha, xmin, s_err, tail_size, ks_pv
#
#
#  # NOTE: do the right/left values have to alternate in output CSV?
#  def tail_stat_zipper(tstat1, tstat2):
#      return [val for pair in zip(tstat1, tstat2) for val in pair]
#
#
#  # TODO: merge this extraction function into get_tail_stats()
#  def get_logl_tstats(daily_log_ratio, daily_log_pv):
#      r_tpl, r_exp, r_logn = daily_log_ratio
#      p_tpl, p_exp, p_logn = daily_log_pv
#      return list((r_tpl, r_exp, r_logn, p_tpl, p_exp, p_logn))


# process given series

def preprocess_series(series):

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
