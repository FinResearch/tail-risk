import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from itertools import product

import yaml
from powerlaw import Fit
from ._plpva import plpva as _plpva


# TODO: remove needless assertions after code is tested

class Analyzer(ABC):

    def __init__(self, ctrl_settings, data_settings):
        # TODO: consider structuring settings objs & attrs better
        self.cs = ctrl_settings
        self.ds = data_settings
        self._set_subcls_spec_props()
        self.outcol_labels = self.__load_output_columns_labels()
        self.results_df = self.__init_results_df()  # TODO: rename to just "results"?

    @abstractmethod
    def _set_subcls_spec_props(self):
        self.output_cfg = None      # str (basename of config file)
        self.output_index = None    # list/tuple (prop from data_settings)
        self.iter_id_keys = None    # iterator

    # # # state INDEPENDENT methods # # #

    def __load_output_columns_labels(self):
        DIR = 'config/output_columns/'  # TODO: improve package/path system
        with open(f'{DIR}/{self.cfg_fname}') as cfg:
            return yaml.load(cfg, Loader=yaml.SafeLoader)

    def __init_results_df(self):
        index = self.output_index
        columns = self.outcol_labels
        df_tail = pd.DataFrame(np.zeros(shape=(len(index), len(columns))),
                               index=index, columns=columns, dtype=float)
        return pd.concat({tail: df_tail for tail in self.ds.tails_to_use})

    # # # state DEPENDENT (or aware) methods # # #

    @abstractmethod
    def _set_curr_input_array(self):
        # NOTE: storage posn into results_df (curr_df_pos) also set here
        pass

    # configure given series to chosen return_type
    def __config_data_by_return_type(self):
        # TODO: add fullname for return_types, ex. {"log": "Log Returns"}
        print(f"You opted for the analysis of {self.ds.return_type} returns")
        # TODO: shove the above into a verbosity level logging

        pt_f = self.curr_input_array[self.ds.tau:]
        pt_i = self.curr_input_array[0: self.ds.len_dates - self.ds.tau]

        if self.ds.return_type == "raw":
            X = pt_f - pt_i
        elif self.ds.return_type == "relative":
            X = pt_f / pt_i - 1.0
        elif self.ds.return_type == "log":
            X = np.log(pt_f/pt_i)
        return X

    # TODO: rewrite this better (more modular, more explicit interface, etc.)
    def _preprocess_curr_input_array(self):
        X = self.__config_data_by_return_type()
        # TODO: std/abs only applies to static when _target == 'full series'
        if self.ds.standardize is True:
            print("I am standardizing your time series")
            X = (X - X.mean())/X.std()
        if self.ds.absolutize is True:
            print("I am taking the absolute value of your time series")
            X = X.abs()
        #  return X
        self.curr_input_array = X

    def __get_xmin(self):
        # TODO: calculate clauset xmin value a priori using lookback
        if self.ds.approach in ("clauset", "manual"):
            xmin = self.ds.xmin_vqty
        elif self.ds.approach == "percentile":
            xmin = np.percentile(self.curr_input_array, self.ds.xmin_vqty)
        return xmin

    def _set_curr_fit_obj(self, tdir):
        X = self.curr_input_array
        data = X if tdir == 'right' else -X
        data = np.nonzero(data)  # NOTE: only keep/use non-zero elements
        discrete = False if self.ds.data_nature == 'continuous' else False
        xmin = self.__get_xmin()
        self.curr_fit = Fit(data, discrete=discrete, xmin=xmin)

    def __get_curr_tail_stats(self):
        alpha, xmin, sigma = (getattr(self.curr_fit.power_law, prop)
                              for prop in ('alpha', 'xmin', 'sigma'))
        abs_len = len(self.curr_input_array[self.curr_input_array >= xmin])
        ks_pv, _ = _plpva(self.curr_input_array, xmin, 'reps',
                          self.ds.plpva_iter, 'silent')
        locs = locals()
        return {vn: locs.get(vn) for vn in self.outcol_labels if vn in locs}

    # TODO: consider moving under DynamicAnalyzer only
    def __get_curr_logl_stats(self):
        logl_stats = {}
        for key, distro in {'tpl': 'truncated_power_law',
                            'exp': 'exponential',
                            'lgn': 'lognormal'}.items():
            R, p = self.curr_fit.distribution_compare('power_law', distro,
                                                      normalized_ratio=True)
            logl_stats[f'R_{key}'] = R
            logl_stats[f'p_{key}'] = p
        return logl_stats

    @abstractmethod
    def _set_curr_rslt_series(self):
        pass

    def _store_partial_results(self, tdir):
        self._set_curr_rslt_series()
        assert len(self.curr_rslt_series) == len(self.outcol_labels)
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        self.results_df.loc[idx, col + (tdir,)].update(self.curr_rslt_series)

    # # # orchestration / driver methods # # #

    # runs single iteration (corresponds to 1 set of input data) of analysis
    def _analyze_next(self):  # TODO: pass iter_id to resume from saved
        self.curr_iter_id = next(self.iter_id_keys)
        self._set_curr_input_array()
        self._preprocess_curr_input_array()  # TODO: group preprocess w/ above?
        for tdir in self.ds.tails_to_use:
            self._set_curr_fit_obj(tdir)
            self._store_partial_results(tdir)

    # publicly exposed method to be called by the user
    def analyze(self):
        while True:
            try:
                self._analyze_next()
            except StopIteration:
                break


class StaticAnalyzer(Analyzer):

    def __init__(self, ctrl_settings, data_settings):
        super(StaticAnalyzer, self).__init__(ctrl_settings, data_settings)
        assert self.ds.approach == 'static'

    def _set_subcls_spec_props(self):
        self.cfg_fname = 'static.yaml'
        self.output_index = self.ds.tickers
        self.iteration_keys = enumerate(self.ds.tickers)

    def _set_curr_input_array(self):
        _, tick = self.curr_iter_id
        self.curr_input_array = self.ds.dbdf[tick].array
        self.curr_df_pos = tick, ()

    # TODO: set curr_rslt_series in __get_curr_tail_stats
    def _set_curr_rslt_series(self):
        self.curr_rslt_series = pd.Series(self.__get_curr_tail_stats())


class DynamicAnalyzer(Analyzer):

    def __init__(self, ctrl_settings, data_settings):
        super(StaticAnalyzer, self).__init__(ctrl_settings, data_settings)
        assert self.ds.approach in ('rolling', 'increasing')

        assert self.ds.lookback is not None
        self.lkb_0 = self.ds.ind_i - self.ds.lookback + 1

    def _set_subcls_spec_props(self):
        self.cfg_fname = 'individual_dynamic.yaml'
        self.output_index = self.ds.anal_dates
        self.iteration_keys = product(enumerate(self.ds.tickers),
                                      enumerate(self.ds.anal_dates))

    def __init_results_df(self):
        df_tick = super(DynamicAnalyzer, self).__init_results_df()
        return pd.concat({tick: df_tick for tick in self.ds.tickers})

    # TODO: consider vectorizing operations on all tickers
    def _set_curr_input_array(self):
        (_, tick), (d, date) = self.curr_iter_id
        d_ind = self.ds.ind_i + d  # TODO: consider only use labels to slice??
        d_lkb = self.lkb_0 + d if self.ds.approach == 'rolling' else self.lkb_0
        self.curr_input_array = self.ds.full_dbdf[tick].iloc[d_lkb:d_ind].array
        self.curr_df_pos = date, (tick,)

    # TODO: set curr_rslt_series in __get_curr_tail_stats,
    #       and call __get_curr_logl_stats here too,
    #       thus removing the need for this method
    def _set_curr_rslt_series(self):
        self.curr_rslt_series = pd.Series({**self.__get_curr_tail_stats(),
                                           **self.__get_curr_logl_stats()})
