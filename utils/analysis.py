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
        self.outcol_labels = self._load_output_columns_labels()
        self.results_df = self._init_results_df()  # TODO: rename as "results"?

    @abstractmethod
    def _set_subcls_spec_props(self):
        self.output_cfg = None      # str (basename of config file)
        self.output_index = None    # list/tuple (prop from data_settings)
        self.iter_id_keys = None    # iterator

    # # # state INDEPENDENT methods # # #

    def _load_output_columns_labels(self):
        DIR = 'config/output_columns/'  # TODO: improve package/path system
        with open(f'{DIR}/{self.cfg_fname}') as cfg:
            return yaml.load(cfg, Loader=yaml.SafeLoader)

    def _init_results_df(self):
        index = self.output_index
        columns = self.outcol_labels
        df_tail = pd.DataFrame(np.zeros(shape=(len(index), len(columns))),
                               index=index, columns=columns, dtype=float)
        return pd.concat({t: df_tail for t in self.ds.tails_to_use}, axis=1)

    # # # state DEPENDENT (or aware) methods # # #

    @abstractmethod
    def _set_curr_input_array(self):
        # NOTE: storage posn into results_df (curr_df_pos) also set here
        pass

    # configure given series to chosen returns_type
    def __config_data_by_returns_type(self, data_array):
        # TODO: shove below printing into verbosity logging
        print(f"You opted for the analysis of {self.ds.returns_type} returns")
        pt_i = data_array[:-self.ds.tau]
        pt_f = data_array[self.ds.tau:]
        if self.ds.returns_type == "raw":
            X = pt_f - pt_i
        elif self.ds.returns_type == "relative":
            X = pt_f / pt_i - 1.0
        elif self.ds.returns_type == "log":
            X = np.log(pt_f/pt_i)
        return X

    # TODO: rewrite this better (more modular, more explicit interface, etc.)
    def _preprocess_data_array(self, data_array):
        X = self.__config_data_by_returns_type(data_array)
        # TODO: std/abs only applies to static when _target == 'full series'
        if self.ds.standardize is True:
            print("I am standardizing your time series")
            X = (X - X.mean())/X.std()
        if self.ds.absolutize is True:
            print("I am taking the absolute value of your time series")
            X = X.abs()
        return X

    def __get_xmin(self):
        # TODO: calculate clauset xmin value a priori using lookback
        if self.ds.xmin_rule in ("clauset", "manual"):
            xmin = self.ds.xmin_vqty
        elif self.ds.xmin_rule == "percentile":
            xmin = np.percentile(self.curr_input_array, self.ds.xmin_vqty)
        return xmin

    def _set_curr_fit_obj(self, tdir):
        X = self.curr_input_array
        data = X if tdir == 'right' else -X
        data = data[np.nonzero(data)]  # NOTE: only keep/use non-zero elements
        discrete = False if self.ds.data_nature == 'continuous' else False
        xmin = self.__get_xmin()
        self.curr_fit = Fit(data, discrete=discrete, xmin=xmin)

    #  def __get_curr_tail_stats(self):
    def _get_curr_tail_stats(self):
        alpha, xmin, sigma = (getattr(self.curr_fit.power_law, prop)
                              for prop in ('alpha', 'xmin', 'sigma'))
        abs_len = len(self.curr_input_array[self.curr_input_array >= xmin])
        ks_pv, _ = _plpva(self.curr_input_array, xmin, 'reps',
                          self.ds.plpva_iter, 'silent')
        locs = locals()
        return {vn: locs.get(vn) for vn in self.outcol_labels if vn in locs}

    # TODO: consider moving under DynamicAnalyzer only
    #  def __get_curr_logl_stats(self):
    def _get_curr_logl_stats(self):
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
        #  print(self.results_df)
        #  print(self.results_df.info())
        #  print(self.results_df.shape)


class StaticAnalyzer(Analyzer):

    def __init__(self, ctrl_settings, data_settings):
        super(StaticAnalyzer, self).__init__(ctrl_settings, data_settings)
        assert self.ds.approach == 'static'

    def _set_subcls_spec_props(self):
        self.cfg_fname = 'static.yaml'
        self.output_index = self.ds.tickers
        self.iter_id_keys = enumerate(self.ds.tickers)

    def _set_curr_input_array(self):
        _, tick = self.curr_iter_id
        self.curr_df_pos = tick, ()
        data_array = self.ds.dbdf[tick].array
        self.curr_input_array = self._preprocess_data_array(data_array)

    # TODO: set curr_rslt_series in __get_curr_tail_stats
    def _set_curr_rslt_series(self):
        self.curr_rslt_series = pd.Series(self.__get_curr_tail_stats())
        # FIXME: self.__get_curr_tail_stats above DNE due to name mangling


class DynamicAnalyzer(Analyzer):

    def __init__(self, ctrl_settings, data_settings):
        super(DynamicAnalyzer, self).__init__(ctrl_settings, data_settings)
        assert self.ds.approach in ('rolling', 'increasing')

        assert self.ds.lookback is not None
        self.lkb_0 = self.ds.ind_i - self.ds.lookback + 1

    def _set_subcls_spec_props(self):
        self.cfg_fname = 'individual_dynamic.yaml'
        self.output_index = self.ds.anal_dates
        self.iter_id_keys = product(enumerate(self.ds.tickers),
                                    enumerate(self.ds.anal_dates))
                                    #  start=self.ds.ind_i))

    #  def __init_results_df(self):
    def _init_results_df(self):
        df_tick = super(DynamicAnalyzer, self)._init_results_df()
        return pd.concat({tick: df_tick for tick in self.ds.tickers}, axis=1)

    # TODO: consider vectorizing operations on all tickers
    def _set_curr_input_array(self):
        (_, tick), (d, date) = self.curr_iter_id
        self.curr_df_pos = date, (tick,)
        d_ind = self.ds.ind_i + d  # TODO: consider only use labels to slice??
        d_lkb = self.lkb_0 + d if self.ds.approach == 'rolling' else self.lkb_0
        print(f"analyzing time series for ticker '{tick}' "  # TODO: VerboseLog
              f"b/w [{self.ds.full_dates[d_lkb]}, {date}]")
        data_array = self.ds.full_dbdf[tick].iloc[d_lkb:d_ind].array
        self.curr_input_array = self._preprocess_data_array(data_array)

    # TODO: override self._get_curr_tail_stats() for DynamicAnalyzer,
    #       set self.curr_rslt_series & call __get_curr_logl_stats() there,
    #       and call in _store_partial_results(), thus remove need for this mtd
    def _set_curr_rslt_series(self):
        self.curr_rslt_series = pd.Series({**self._get_curr_tail_stats(),
                                           **self._get_curr_logl_stats()})
