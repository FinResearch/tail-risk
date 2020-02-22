import numpy as np
from settings import settings as s

import powerlaw as pl


# Common functions in both if-branches

def _get_xmin(data, xmin=None):

    xmin_rule, xmin_val = s.xmin_inputs

    if xmin is not None:  # NOTE: use this for Rolling
        return xmin

    if xmin_rule == "clauset":
        xmin = None
    elif xmin_rule == "manual":
        xmin = xmin_val
    elif xmin_rule == "percentile":
        xmin = np.percentile(data, xmin_val)

    return xmin


def init_fit_obj(data, xmin=None):

    # NOTE: only keep/use non-zero elements
    data1 = np.array(data)[np.nonzero(data)]

    # data_nature is one of ['Discrete', 'Continuous']
    discrete = False if s.data_nature == 'continuous' else True

    xmin = _get_xmin(data, xmin)

    return pl.Fit(data1, discrete=discrete, xmin=xmin)


def get_fit(data):

    fit = init_fit_obj(data)

    # TODO: test standardization branch
    if s.standardize and s.standardize_target == "tail":
        xmin = fit.power_law.xmin
        fit = standardize_tail(data, xmin)

    return fit


# TODO: test standardization function
def standardize_tail(tail_data, xmin):
    print("I am standardizing your tail")
    S = np.array(tail_data)
    S = S[S >= xmin]
    m = np.mean(S)
    v = np.std(S)
    X = (S - m) / v

    if s.absolutize and s.absz_target == "tail":
        X = absolutize_tail(X)

    return init_fit_obj(X, s.xmin_rule, np.min(X))


# TODO: test absolutize function
def absolutize_tail(tail_data):
    print("I am taking the absolute value of your tail")
    #  lab = "|" + lab + "|"
    return np.abs(tail_data)


# configure given series to chosen return_type

def cofig_series(series):

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
