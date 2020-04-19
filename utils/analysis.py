import numpy as np
import pandas as pd
# NOTE: if pd.Series replaced, then module no longer depends on pandas

from abc import ABC, abstractmethod
from itertools import product

from powerlaw import Fit  # TODO: consider import entire module?
from ._plpva import plpva as _plpva
from .configure import DataConfigurer
from .results import Results

import sys  # TODO: remove after debugging uses done
from os import getpid  # TODO: remove after debugging uses done
from multiprocessing import Pool  # TODO: import as mp?


class _Analyzer(ABC):

    def __init__(self, settings):
        self.sc = settings.ctrl
        self.sd = settings.data
        self.sa = settings.anal

        self.cfg = DataConfigurer(settings)
        self.res = Results(settings)

    # # # state DEPENDENT (or aware) methods # # #

    def _log_curr_iter(self):
        # TODO: factor out repetitive log? (static: date, dynamic: group_label)
        gtyp, *date, tail = self.curr_iter_id
        grp_tail_log = (f"Analyzing {tail.name.upper()} tail of time series "
                        f"for {self.sd.grouping_type.title()} '{gtyp}' ")
        if bool(date):  # dynamic approach
            df = date[0]
            di = self.sa.get_dyn_lbd(df)
        else:           # static approach
            di, df = self.sd.date_i, self.sd.date_f
        date_log = f"b/w [{di}, {df}]"
        print(grp_tail_log + date_log)

    @abstractmethod
    # TODO --> def _slice_dbdf_data(self):
    def _set_curr_input_array(self):
        # NOTE: storage posn into results_df (curr_df_pos) also set here
        pass

    def __get_xmin(self):
        if self.sa.xmin_rule in {"clauset", "manual"}:
            xmin = self.sa.xmin_qnty  # ie. {None, user-input-ℝ} respectively
        elif self.sa.xmin_rule == "percent":
            xmin = np.percentile(self.curr_input_array, self.sa.xmin_qnty)
        elif self.sa.xmin_rule in {"file", "average"}:
            assert self.sa.use_dynamic,\
                ("static approach does NOT currently support passing "
                 "xmin data by file")  # TODO: add file support for -a static?
            grp, date, tail = self.curr_iter_id
            tst = self.sa.tst_map[tail]
            win_size = (f' {self.sa.rws}'
                        if self.sa.xmin_rule == 'average' else '')
            xmin = self.sa.xmin_qnty.loc[date, f"{tst} {grp}{win_size}"]
        else:
            raise AttributeError("this should never be reached!")
        return xmin

    def _fit_curr_data(self):
        data = self.curr_input_array
        data = data[np.nonzero(data)]  # only use non-zero elements to do Fit
        xmin = self.__get_xmin()
        self.curr_fit = Fit(data=data, xmin=xmin,
                            discrete=self.sa.fit_discretely)

    def _get_curr_tail_stats(self):
        alpha, xmin, sigma = (getattr(self.curr_fit.power_law, prop)
                              for prop in ('alpha', 'xmin', 'sigma'))
        abs_len = sum(self.curr_input_array >= xmin)
        if self.sa.run_ks_test is True:
            # TODO: try compute ks_pv using MATLAB engine & module, and time
            ks_pv, _ = _plpva(self.curr_input_array, xmin, 'reps',
                              self.sa.ks_iter, 'silent')
        locs = locals()
        return {(stat, ''): locs.get(stat) for stat, _ in
                self.sd.stats_collabs if stat in locs}

    def _store_partial_results(self):
        curr_tstat_series = pd.Series(self._get_curr_tail_stats())
        assert len(curr_tstat_series) == len(self.sd.stats_collabs)
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        self.res.df.loc[idx, col].update(curr_tstat_series)
        # TODO: consider using pd.DataFrame.replace(, inplace=True) instead
        # TODO: can also order stats results first, then assign to DF row

    def __get_iter_results(self):  # return results tuple for one tail
        # TODO: use np.ndarray instead of pd.Series (wasteful) --> order later
        curr_tstat_series = pd.Series(self._get_curr_tail_stats())
        assert len(curr_tstat_series) == len(self.sd.stats_collabs)
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        return (idx, col), curr_tstat_series  # (df_posn, df_value) of results

    # # # orchestration / driver methods # # #

    # convenience wrapper to keep things tidy
    def _run_curr_iter_fitting(self):
        self._log_curr_iter()
        self._set_curr_input_array()
        self._fit_curr_data()

    # runs analysis on data ID'd by the next iteration of the stateful iterator
    def _analyze_next(self):  # TODO: combine _analyze_next & _analyze_iter??
        self.curr_iter_id = next(self.iter_id_keys)  # set in subclasses
        self._run_curr_iter_fitting()
        self._store_partial_results()

    # runs analysis from start to finish, in 1-process + single-threaded mode
    def analyze_sequential(self):
        while True:
            try:
                self._analyze_next()
            except StopIteration:
                break

    # runs analysis for one iteration of analysis given arbitrary iter_id
    def _analyze_iter(self, iter_id):  # NOTE: use this to resume computation
        print(f"### DEBUG: PID {getpid()} analyzing iter {iter_id}", file=sys.stderr)
        self.curr_iter_id = iter_id
        self._run_curr_iter_fitting()
        return self.__get_iter_results()

    # runs analysis in multiprocessing mode
    def analyze_multiproc(self):
        # TODO: https://stackoverflow.com/a/52596590/5437918 (use shared DBDFs)
        iter_id_keys = tuple(self.iter_id_keys)

        # TODO: look into Queue & Pipe for sharing data
        with Pool(processes=self.sc.nproc) as pool:
            # TODO checkout .map alternatives: .imap, .map_async, etc.
            restup_ls = [restup for restup in  # TODO: optimize chunksize below
                         pool.map(self._analyze_iter, iter_id_keys)]

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
        #       - use .title() on all index labels, then write to file
        self.res.prettify_df()
        return self.res.df


class StaticAnalyzer(_Analyzer):

    def __init__(self, settings):
        super().__init__(settings)
        assert not self.sa.use_dynamic
        self.iter_id_keys = product(self.sd.grouping_labs,
                                    self.sa.tails_to_anal)
        # TODO: cache preproc'd data arr above so 2nd tail need not recompute;
        # NOTE: consider make tails_to_anal as 1st factor in itertools.product,
        #       so data array for one set of tail are all calculated first

    def _set_curr_input_array(self):  # TODO: pass curr_iter_id as arg???
        lab, tail = self.curr_df_pos = self.curr_iter_id
        self.curr_input_array = self.cfg.get_data(lab) * tail.value


class DynamicAnalyzer(_Analyzer):

    def __init__(self, settings):
        super().__init__(settings)
        assert self.sa.use_dynamic
        self.iter_id_keys = product(self.sd.grouping_labs,
                                    self.sd.anal_dates,
                                    self.sa.tails_to_anal)
        # TODO: see TODO & NOTE regarding Tail in __init__ of StaticAnalyzer
        self._distros_to_compare = {'ll_tpl': 'truncated_power_law',
                                    'll_exp': 'exponential',
                                    'll_lgn': 'lognormal'}

    # TODO: consider vectorizing operations on all tickers
    def _set_curr_input_array(self):  # TODO: pass curr_iter_id as arg???
        sub, date, tail = self.curr_iter_id
        self.curr_df_pos = date, (sub, tail)
        self.curr_input_array = self.cfg.get_data((sub, date)) * tail.value

    def _get_curr_logl_stats(self):
        # compute (R, p) using powerlaw.Fit.distribution_compare
        logl_stats = {key:
                      {stat: val for stat, val in
                       zip(('R', 'p'),
                           self.curr_fit.distribution_compare(
                               'power_law', distro,
                               normalized_ratio=True))}
                      for key, distro in self._distros_to_compare.items()}

        return {(key, stat): logl_stats.get(key, {}).get(stat) for
                key, stat in self.sd.stats_collabs if key.startswith('ll_')}

    # TODO: add getting xmin_today data when doing group tail analysis
    def _get_curr_tail_stats(self):
        tail_stats = super()._get_curr_tail_stats()
        logl_stats = (self._get_curr_logl_stats()
                      if self.sa.compare_distros else {})
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
