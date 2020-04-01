import numpy as np
import pandas as pd
# NOTE: if pd.Series replaced, then module no longer depends on pandas

from abc import ABC, abstractmethod
from itertools import product

from powerlaw import Fit  # TODO: consider import entire module?
from ._plpva import plpva as _plpva
from .results import Results

from os import getpid  # TODO: remove after debugging uses done
from multiprocessing import Pool  # TODO: import as mp?


class _Analyzer(ABC):

    def __init__(self, settings):
        self.sc = settings.ctrl
        self.sd = settings.data
        self.sa = settings.anal

        self.res = Results(settings)

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
        print("You opted for the analysis of "
              f"{self.sa.returns_type} returns")
        pt_i = data_array[:-self.sa.tau]  # ASK/TODO: if anal_freq > 1, then tau of 1 isn't really a 'daily' lag
        pt_f = data_array[self.sa.tau:]
        if self.sa.returns_type == "raw":
            X = pt_f - pt_i
        elif self.sa.returns_type == "relative":
            X = pt_f / pt_i - 1.0
        elif self.sa.returns_type == "log":
            X = np.log(pt_f/pt_i)
        return X

    # TODO: rewrite this better (more modular, more explicit interface, etc.)
    def _preprocess_data_array(self, data_array):
        X = self._config_data_by_returns_type(data_array)
        # TODO: std/abs only applies to static when _target == 'full series'
        if self.sa.standardize is True:
            print("I am standardizing your time series")
            X = (X - X.mean())/X.std()
            # TODO/ASK/NOTE: np.std/np.ndarray.std uses ddof=0;
            #                pd.DataFrame.rolling().std uses ddof=1
            # i.e. each input considered a true population or just a sample?
        if self.sa.absolutize is True:
            print("I am taking the absolute value of your time series")
            X = X.abs()
        return X

    def __get_xmin(self):
        # TODO: calculate clauset xmin value a priori using lookback
        if self.sa.xmin_rule in {"clauset", "manual"}:
            xmin = self.sa.xmin_vqty
        elif self.sa.xmin_rule == "percentile":
            # NOTE: take percentile of non-filtered for nonzeros series,
            #       but after potential normalization
            xmin = np.percentile(self.curr_input_array, self.sa.xmin_vqty)
        elif self.sa.xmin_rule == "average":
            pass
        else:
            raise AttributeError(f"invalid xmin-rule: {self.sa.xmin_rule}")
        return xmin

    def _calc_curr_fit_obj(self, tail):
        data = self.curr_input_array * tail.value  # tail.value = ±1 for R/L
        # TODO: can filter for nonzeros before doing above multiplication
        data = data[np.nonzero(data)]  # only keep/use non-zero elements
        xmin = self.__get_xmin()  # outsource this to settings.py
        self.curr_fit = Fit(data=data, xmin=xmin,  # xmin=self.sa.xmin,
                            discrete=self.sa.fit_discretely)

    def _get_curr_tail_stats(self):
        alpha, xmin, sigma = (getattr(self.curr_fit.power_law, prop)
                              for prop in ('alpha', 'xmin', 'sigma'))
        abs_len = len(self.curr_input_array[self.curr_input_array >= xmin])
        if self.sa.run_ks_test is True:
            # TODO: try compute ks_pv using MATLAB engine & module, and time
            ks_pv, _ = _plpva(self.curr_input_array, xmin, 'reps',
                              self.sa.ks_iter, 'silent')
        locs = locals()
        return {(stat, ''): locs.get(stat) for stat, _ in
                self.sd.stats_collabs if stat in locs}

    def _store_partial_results(self, tail):
        curr_tstat_series = pd.Series(self._get_curr_tail_stats())
        # TODO: remove needless assertion stmt(s) after code is well-tested
        assert len(curr_tstat_series) == len(self.sd.stats_collabs)
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        self.res.df.loc[idx, col + (tail,)].update(curr_tstat_series)
        # TODO: consider using pd.DataFrame.replace(, inplace=True) instead

    def __get_tdir_iter_restup(self, tail):  # retrn results tuple for one tail
        # TODO: use np.ndarray instead of pd.Series (wasteful) --> order later
        curr_tstat_series = pd.Series(self._get_curr_tail_stats())
        assert len(curr_tstat_series) == len(self.sd.stats_collabs)
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        return (idx, col + (tail,)), curr_tstat_series

    # # # orchestration / driver methods # # #

    # runs analysis on data ID'd by the next iteration of the stateful iterator
    def _analyze_next(self):
        self.curr_iter_id = next(self.iter_id_keys)  # set in subclasses
        self._set_curr_input_array()  # 'input' as in input to powerlaw.Fit
        for tail in self.sa.tails_to_anal:
            self._calc_curr_fit_obj(tail)
            self._store_partial_results(tail)

    # runs analysis from start to finish, in 1-process + single-threaded mode
    def analyze_sequential(self):
        while True:
            try:
                self._analyze_next()
            except StopIteration:
                break

    # runs analysis for one iteration of analysis given arbitrary iter_id
    def _analyze_iter(self, iter_id):  # NOTE: use this to resume computation
        print(f"### DEBUG: PID {getpid()} analyzing iter '{iter_id}'")

        self.curr_iter_id = iter_id
        self._set_curr_input_array()  # 'input' as in input to powerlaw.Fit

        iter_restups = []  # results tuple(s) for single iteration of input
        for tail in self.sa.tails_to_anal:
            self._calc_curr_fit_obj(tail)
            iter_restups.append(self.__get_tdir_iter_restup(tail))
        return iter_restups

    # runs analysis in multiprocessing mode
    def analyze_multiproc(self):
        # TODO: https://stackoverflow.com/a/52596590/5437918 (use shared DBDFs)
        iter_id_keys = tuple(self.iter_id_keys)

        # TODO: look into Queue & Pipe for sharing data
        with Pool(processes=self.sc.nproc) as pool:
            # TODO checkout .map alternatives: .imap, .map_async, etc.
            restup_ls = [restup for iter_restups in  # TODO: optimize chunksize
                         pool.map(self._analyze_iter, iter_id_keys)
                         for restup in iter_restups]
        assert len(restup_ls) == len(self.sa.tails_to_anal)*len(iter_id_keys)

        # TODO: update results_df more efficiently, ex. pd.DataFrame.replace(),
        #       np.ndarray, etc.; see TODO note under __get_tdir_iter_restup)
        for restup in restup_ls:
            (idx, col), res = restup  # if use '+' NOTE that DFs init'd w/ NaNs
            self.res.df.loc[idx, col].update(res)

    # top-level convenience method that autodetects how to run tail analysis
    def analyze(self):
        nproc = self.sc.nproc
        # TODO: add other conditions for analyze_sequential (ex. -a static)
        if nproc == 1:
            self.analyze_sequential()
        elif nproc > 1:
            self.analyze_multiproc()
        else:
            raise TypeError(f'Cannot perform analysis with {nproc} processes')

    def get_resdf(self):
        # TODO: final clean ups of DF for presentation:
        #       - use df.columns = df.columns.droplevel() to remove unused lvls
        #       - use .title() on all index labels, then write to file
        return self.res.df


class StaticAnalyzer(_Analyzer):

    def __init__(self, settings):
        super().__init__(settings)
        self.iter_id_keys = iter(self.sd.grouping_labs)
        assert not self.sa.use_dynamic

    def _set_curr_input_array(self):  # TODO:consider pass curr_iter_id as arg?
        lab = self.curr_iter_id
        self.curr_df_pos = lab, ()
        # TODO: move logging of DATE RANGE out of this repeatedly called method
        print(f"Analyzing time series for {self.sd.grouping_type.title()} "
              f"'{lab}' b/w [{self.sd.date_i}, {self.sd.date_f}]")
        # TODO optimize below: when slice is pd.Series, no need to .flatten()
        data_array = self.sd.static_dbdf[lab].to_numpy().flatten()
        # TODO: factor below _preprocess_data_array call into ABC??
        self.curr_input_array = self._preprocess_data_array(data_array)


class DynamicAnalyzer(_Analyzer):

    def __init__(self, settings):
        super().__init__(settings)
        assert self.sa.use_dynamic
        self.iter_id_keys = product(iter(self.sd.grouping_labs),
                                    enumerate(self.sd.anal_dates,
                                              start=self.sd.date_i_idx))
        self.lkb_off = self.sa.lookback - 1  # ASK/TODO: offset 0 or 1 day?
        self.lkb_0 = self.sd.date_i_idx - self.lkb_off

        self._distros_to_compare = {'ll_tpl': 'truncated_power_law',
                                    'll_exp': 'exponential',
                                    'll_lgn': 'lognormal'}

    # TODO: consider vectorizing operations on all tickers
    def _set_curr_input_array(self):  # TODO:consider pass curr_iter_id as arg?
        sub, (d, date) = self.curr_iter_id
        self.curr_df_pos = date, (sub,)
        d0 = (d - self.lkb_off if self.sa.approach == 'rolling'  # TODO: determine this in settings.py?? --> create d0_generator??
              else self.lkb_0)  # TODO: calc all needed dates in settings.py??
        # TODO: move logging of LABEL out of this repeatedly called method
        print(f"Analyzing time series for {self.sd.grouping_type.title()} "
              f"'{sub}' b/w [{self.sd.full_dates[d0]}, {date}]")  # TODO:-VVV
        data_array = self.sd.dynamic_dbdf[sub].iloc[d0: d].\
            to_numpy().flatten()
        # TODO/ASK: for group input array: slice dbdf --> .to_numpy().flatten()
        #           confirm flattening order does not matter
        self.curr_input_array = self._preprocess_data_array(data_array)

    def _get_curr_logl_stats(self):  # ASK/TODO: logl_stats unneeded in static?
        # compute (R, p) using powerlaw.Fit.distribution_compare
        logl_stats = {key:
                      {stat: val for stat, val in
                       zip(('R', 'p'),
                           self.curr_fit.distribution_compare(
                               'power_law',
                               distro,
                               normalized_ratio=True))
                       }
                      for key, distro in self._distros_to_compare.items()}

        return {(key, stat): logl_stats.get(key, {}).get(stat) for
                key, stat in self.sd.stats_collabs if key.startswith('ll_')}

    # TODO: add getting xmin_today data when doing group tail analysis
    def _get_curr_tail_stats(self):
        tail_stats = super()._get_curr_tail_stats()
        logl_stats = self._get_curr_logl_stats()
        return {**tail_stats, **logl_stats}


# wrapper func: instantiate correct Analyzer type and run tail analysis
def analyze_tail(settings):
    Analyzer = DynamicAnalyzer if settings.anal.use_dynamic else StaticAnalyzer
    analyzer = Analyzer(settings)
    analyzer.analyze()
    results = analyzer.get_resdf()
    print(results)
    print('-' * 100)
    print(results.info())
    print('-' * 100)
