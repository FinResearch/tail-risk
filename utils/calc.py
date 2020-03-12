# TODO: revise function and variable names to be better descriptive

import numpy as np

from powerlaw import Fit
from ._plpva import plpva as _plpva

#  pl_distro_map = {'tpl': "truncated_power_law",
#                   'exp': "exponential",
#                   'logn': "lognormal"}


class Calculator:

    def __init__(self, data_settings):
        self.ds = data_settings
        self.pl_distros = ('truncated_power_law', 'exponential', 'lognormal')

    def __get_xmin(self, xmin=None, data=None, fit=None):
        # NOTE: if explicitly passed xmin to function, then ofc use it
        if xmin is not None:  # NOTE: use this for Rolling xmin_rule in GroupTA
            return xmin

        if self.ds.xmin_rule == "clauset":
            xmin = None
        elif self.ds.xmin_rule == "manual":
            xmin = self.ds.vqty
        elif self.ds.xmin_rule == "percentile":
            if data is None:
                raise TypeError("Need input data vector to calculate "
                                "xmin value for rule 'percentile'")
            xmin = np.percentile(data, self.ds.xmin_vqty)
        elif self.ds.xmin_rule == 'average':
            if data is None or fit is None:
                raise TypeError("Need tail vector and Fit instance to "
                                "calculate xmin value for rule 'average'")
            days, lag = self.ds.xmin_vqty
            #  xmin = np.average(x_min_right_vector[-(days+lags): -lags])
        return xmin

    def _get_fit_obj(self, data, xmin=None):
        # NOTE: only keep/use non-zero elements
        data = np.nonzero(data)  # TODO: confirm data is always of np.ndarray
        discrete = False if self.ds.data_nature == 'continuous' else False
        xmin = self.__get_xmin(xmin=xmin, data=data)
        return Fit(data, discrete=discrete, xmin=xmin)

    def _std_abs_tail(self, data, fit):
        # TODO: need to test standardization function
        # TODO: consider passing in fit.power_law.xmin directly, instead of fit

        print("I am standardizing your tail")
        S = np.array(data)  # TODO:ensure data is np.ndarray so no need to cast
        S = S[S >= fit.power_law.xmin]
        X = (S - S.mean()) / S.std()

        if self.ds.absolutize:  # and sett.abs_target == "tail":
            print("I am taking the absolute value of your tail")
            X = np.abs(X)

        return X

    # configure given series to chosen return_type
    def _config_series(self, series):
        # TODO: add fullname for return_types, ex. {"log": "Log Returns"}
        print(f"You opted for the analysis of the {self.ds.return_type}")

        pt_f = series[self.ds.tau:]
        pt_i = series[0: len(series) - self.ds.tau]

        if self.ds.return_type == "basic":
            X = pt_f - pt_i
        elif self.ds.return_type == "relative":
            X = pt_f / pt_i - 1.0
        elif self.ds.return_type == "log":
            X = np.log(pt_f/pt_i)

        # TODO: std/abs only applies to static when _target == 'full series'
        if self.ds.standardize is True:
            print("I am standardizing your time series")
            X = (X - X.mean())/X.std()

        if self.ds.absolutize is True:
            print("I am taking the absolute value of your time series")
            X = X.abs()

        return X

    def _get_tail_stats(self, fit, series):
        fpl = fit.power_law  # fpl: fitted power law
        alpha = fpl.alpha
        xmin = fpl.xmin
        sigma = fpl.sigma

        abs_len = len(series[series >= xmin])
        ks_pv, _ = _plpva(series, xmin, 'reps', self.ds.plpva_iter, 'silent')

        return [alpha, xmin, sigma, abs_len, ks_pv]

    def _get_logl_tstats(self, fit):
        logl_stats = []
        for pdf in self.pl_distros:
            R, p = fit.distribution_compare("power_law", pdf,
                                            normalized_ratio=True)
            logl_stats.append(R)
            logl_stats.append(p)
        return logl_stats

    def get_results_tup(self, series):
        X = self._config_series(series)

        trl = []  # trl: tails results list
        for tdir in self.ds.tails_used:
            data = X if tdir == 'right' else -X
            fit = self._get_fit_obj(data)

            if (self.ds.approach == 'static' and
                    self.ds.standardize):  # and self.ds.std_target == 'tail'):
                X = self._std_abs_tail(data, fit)
                # FIXME: call below uses manually-passed xmin val for all xmin_rule
                fit = self._get_Fit(X, X.min())
                # NOTE: for above, need X.min() to not apply to clauset & percent??

            tail_stats = self._get_tail_stats(fit, data)
            logl_stats = self._get_logl_tstats(fit)

            trl.append(tail_stats + logl_stats)

        return [stat for stat_tup in zip(*trl) for stat in stat_tup]

    # TODO: make function more efficinet; i.e. avoid making copies
    def get_plot_vecs(self, csv_array_T):

        # TODO: make below row indexing more succinct
        if self.ds.tail_selection == 'both':  # TODO: use len(tails_to_use)?
            alp, sig, abl, aks = (0, 1), (4, 5), (6, 7), (8, 9)
        else:
            alp, sig, abl, aks = 0, 2, 3, 4

        alphas = csv_array_T[np.r_[alp]]
        _sigmas = csv_array_T[np.r_[sig]]
        abs_lens = csv_array_T[np.r_[abl]]
        ks_pvs = csv_array_T[np.r_[aks]]

        bound_deltas = self.ds.alpha_qntl * _sigmas
        up_bounds = alphas + bound_deltas
        low_bounds = alphas - bound_deltas

        # NOTE: rel_len = abs_len / tail_len (known in advance)
        if self.ds.approach == 'rolling':
            tlen = 504
        elif self.ds.approach == 'increasing':
            tlen = None  # FIXME: consider using a generator?
        rel_lens = abs_lens / tlen

        # TODO: consider doing the zip/convolution here?
        return alphas, up_bounds, low_bounds, abs_lens, rel_lens, ks_pvs
