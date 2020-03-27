import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from itertools import product

from powerlaw import Fit  # TODO: consider import entire module?
from ._plpva import plpva as _plpva

from os import getpid  # TODO: remove after debugging uses done
from multiprocessing import Pool


class Analyzer(ABC):

    def __init__(self, ctrl_settings, data_settings):
        # TODO: consider structuring settings objs & attrs better
        self.cs = ctrl_settings
        self.ds = data_settings
        self._set_subcls_iter_props()
        self.results = self._init_results_df()

    # # # state INDEPENDENT methods # # #

    @abstractmethod
    def _set_subcls_iter_props(self):
        # properties initialized below are used by methods defined in this ABC
        self.output_index = None    # list/tuple (prop from data_settings)
        self.iter_id_keys = None    # iterator

    def _init_results_df(self):
        index = self.output_index
        columns = self.ds.outcol_labels
        df_tail = pd.DataFrame(np.zeros(shape=(len(index), len(columns))),
                               index=index, columns=columns, dtype=float)
        return pd.concat({t: df_tail for t in self.ds.tails_to_use}, axis=1)
        # TODO: set index (both columns & row-index) names property

    # # # state DEPENDENT (or aware) methods # # #

    @abstractmethod
    # TODO --> def _slice_dbdf_data(self):
    def _set_curr_input_array(self):
        # NOTE: storage posn into results_df (curr_df_pos) also set here
        pass

    # configure given series to chosen returns_type
    # TODO: can do this all beforehand on entire dbdf, instead of on each iter
    def _config_data_by_returns_type(self, data_array):
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
        X = self._config_data_by_returns_type(data_array)
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
        if self.ds.xmin_rule in {"clauset", "manual"}:
            xmin = self.ds.xmin_vqty
        elif self.ds.xmin_rule == "percentile":
            xmin = np.percentile(self.curr_input_array, self.ds.xmin_vqty)
        return xmin

    def _set_curr_fit_obj(self, tdir):
        data = self.curr_input_array * {'right': +1, 'left': -1}[tdir]
        data = data[np.nonzero(data)]  # NOTE: only keep/use non-zero elements
        discrete = False if self.ds.data_nature == 'continuous' else False
        xmin = self.__get_xmin()
        self.curr_fit = Fit(data=data, discrete=discrete, xmin=xmin)

    def _get_curr_tail_stats(self):
        alpha, xmin, sigma = (getattr(self.curr_fit.power_law, prop)
                              for prop in ('alpha', 'xmin', 'sigma'))
        abs_len = len(self.curr_input_array[self.curr_input_array >= xmin])
        if self.cs.ks_flag is True:  # TODO:add simlar skip opt for logl_stats?
            ks_pv, _ = _plpva(self.curr_input_array, xmin, 'reps',
                              self.ds.ks_iter, 'silent')
            # TODO: try compute ks_pv using MATLAB engine & module, and time
        locs = locals()
        return {vr: locs.get(vr) for vr in self.ds.outcol_labels if vr in locs}

    def _store_partial_results(self, tdir):
        curr_tstat_series = pd.Series(self._get_curr_tail_stats())
        # TODO: remove needless assertion stmt(s) after code is well-tested
        assert len(curr_tstat_series) == len(self.ds.outcol_labels)
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        self.results.loc[idx, col + (tdir,)].update(curr_tstat_series)
        # TODO: consider using pd.DataFrame.replace(, inplace=True) instead

    def __get_tdir_iter_restup(self, tdir):  # retrn results tuple for one tail
        # TODO: use np.ndarray instead of pd.Series (wasteful) --> order later
        curr_tstat_series = pd.Series(self._get_curr_tail_stats())
        assert len(curr_tstat_series) == len(self.ds.outcol_labels)
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        return (idx, col + (tdir,)), curr_tstat_series

    # # # orchestration / driver methods # # #

    # runs analysis on data ID'd by the next iteration of the stateful iterator
    def _analyze_next(self):  # TODO: pass iter_id to resume from saved
        self.curr_iter_id = next(self.iter_id_keys)
        self._set_curr_input_array()  # 'input' as in input to powerlaw.Fit
        for tdir in self.ds.tails_to_use:
            self._set_curr_fit_obj(tdir)
            self._store_partial_results(tdir)

    # runs analysis from start to finish, in 1-process + single-threaded mode
    def analyze_sequential(self):
        while True:
            try:
                self._analyze_next()
            except StopIteration:
                break

    # runs analysis for one iteration of analysis given arbitrary iter_id
    def _analyze_iter(self, iter_id):
        print(f"### DEBUG: PID {getpid()} analyzing iter '{iter_id}'")

        self.curr_iter_id = iter_id
        self._set_curr_input_array()  # 'input' as in input to powerlaw.Fit

        iter_restups = []  # results tuple(s) for single iteration of input
        for tdir in self.ds.tails_to_use:
            self._set_curr_fit_obj(tdir)
            iter_restups.append(self.__get_tdir_iter_restup(tdir))
        return iter_restups

    # runs analysis in multiprocessing mode
    def analyze_multiproc(self):
        # TODO: https://stackoverflow.com/a/52596590/5437918 (use shared DBDFs)
        iter_id_keys = tuple(self.iter_id_keys)

        # TODO: look into Queue & Pipe for sharing data
        with Pool(processes=self.cs.nproc) as pool:
            # TODO checkout .map alternatives: .imap, .map_async, etc.
            restup_ls = [restup for iter_restups in  # TODO: optimize chunksize
                         pool.map(self._analyze_iter, iter_id_keys)
                         for restup in iter_restups]
        assert len(restup_ls) == len(iter_id_keys)*len(self.ds.tails_to_use)

        # TODO: update results_df more efficiently, ex. pd.DataFrame.replace(),
        #       np.ndarray, etc.; see TODO note under __get_tdir_iter_restup)
        for restup in restup_ls:
            (idx, col), res = restup
            self.results.loc[idx, col].update(res)

    # top-level convenience method that autodetects how to run tail analysis
    def analyze(self):
        nproc = self.cs.nproc
        # TODO: add other OR conds for analyze_sequential anal (ex. -a static)
        if nproc == 1:
            self.analyze_sequential()
        elif nproc > 1:
            self.analyze_multiproc()
        else:
            raise TypeError(f'Cannot perform analysis with {nproc} processes')


class StaticAnalyzer(Analyzer):

    def __init__(self, ctrl_settings, data_settings):
        super(StaticAnalyzer, self).__init__(ctrl_settings, data_settings)
        assert self.ds.approach == 'static'

    def _set_subcls_iter_props(self):
        self.output_index = self.ds.tickers_grouping
        self.iter_id_keys = iter(self.ds.tickers_grouping)

    def _set_curr_input_array(self):  # TODO:consider pass curr_iter_id as arg?
        lab = self.curr_iter_id
        self.curr_df_pos = lab, ()
        # TODO: move logging of DATE RANGE out of this repeatedly called method
        print(f"analyzing time series for {self.ds.group_type_label} '{lab}' "
              f"b/w [{self.ds.date_i}, {self.ds.date_f}]")  # TODO: VerboseLog
        # TODO optimize below: when slice is pd.Series, no need to .flatten()
        data_array = self.ds.static_dbdf[lab].to_numpy().flatten()
        # TODO: factor below _preprocess_data_array call into ABC??
        self.curr_input_array = self._preprocess_data_array(data_array)


class DynamicAnalyzer(Analyzer):

    def __init__(self, ctrl_settings, data_settings):
        super(DynamicAnalyzer, self).__init__(ctrl_settings, data_settings)
        assert self.ds.approach in {'rolling', 'increasing'}
        self.lkb_off = self.ds.lookback - 1  # ASK/TODO: offset 0 or 1 day?
        self.lkb_0 = self.ds.date_i_idx - self.lkb_off

        self._distros_to_compare = {'tpl': 'truncated_power_law',
                                    'exp': 'exponential',
                                    'lgn': 'lognormal'}

    def _set_subcls_iter_props(self):
        self.output_index = self.ds.anal_dates
        self.iter_id_keys = product(iter(self.ds.tickers_grouping),
                                    enumerate(self.ds.anal_dates,
                                              start=self.ds.date_i_idx))

    def _init_results_df(self):
        df_sub = super(DynamicAnalyzer, self)._init_results_df()
        return pd.concat({sub: df_sub for sub in self.ds.tickers_grouping},
                         axis=1)  # TODO: set column index level names

    # TODO: consider vectorizing operations on all tickers
    def _set_curr_input_array(self):  # TODO:consider pass curr_iter_id as arg?
        sub, (d, date) = self.curr_iter_id
        self.curr_df_pos = date, (sub,)
        d0 = d - self.lkb_off if self.ds.approach == 'rolling' else self.lkb_0
        # TODO: move logging of LABEL out of this repeatedly called method
        print(f"analyzing time series for {self.ds.group_type_label} '{sub}' "
              f"b/w [{self.ds.full_dates[d0]}, {date}]")
        data_array = self.ds.dynamic_dbdf[sub].iloc[d0: d].to_numpy().flatten()
        # TODO/ASK: for group input array: slice dbdf --> .to_numpy().flatten()
        #           confirm flattening order does not matter
        self.curr_input_array = self._preprocess_data_array(data_array)

    def _get_curr_logl_stats(self):  # ASK/TODO: logl_stats unneeded in static?
        logl_stats = {}
        for key, distro in self._distros_to_compare.items():
            R, p = self.curr_fit.distribution_compare('power_law', distro,
                                                      normalized_ratio=True)
            logl_stats[f'R_{key}'] = R  # TODO: store R, p together as (R, p)?
            logl_stats[f'p_{key}'] = p  # even better: store as 2-lvl'd sub-DF
        return logl_stats

    # TODO: add getting xmin_today data when doing group tail analysis
    def _get_curr_tail_stats(self):
        tail_stats = super(DynamicAnalyzer, self)._get_curr_tail_stats()
        logl_stats = self._get_curr_logl_stats()
        return {**tail_stats, **logl_stats}


# wrapper func: instantiate correct Analyzer type and run tail analysis
def analyze_tail(ctrl_settings, data_settings):
    cs, ds = ctrl_settings, data_settings
    Analyzer = StaticAnalyzer if ds.approach == 'static' else DynamicAnalyzer
    analyzer = Analyzer(cs, ds)
    analyzer.analyze()
    print(analyzer.results)
