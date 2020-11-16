import numpy as np
import pandas as pd
# NOTE: if pd.Series replaced, then module no longer depends on pandas

import statistics as st
import scipy.stats

from abc import ABC, abstractmethod
from itertools import product

from powerlaw import Fit
from ._plpva import plpva
from .returns import Returns
from .results import Results

import sys  # TODO: remove sys module & os.getpid after debugging uses done
from os import getpid
from multiprocessing import Pool


class _Analyzer(ABC):

    def __init__(self, settings):
        self.sc = settings.ctrl
        self.sd = settings.data
        self.sa = settings.anal

        # TODO: factor setting of these boolean flags into own method
        if self.sa.txmin_map:
            self._use_pct_file = any('PCT' in col_hdr for col_hdr
                                     in self.sa.txmin_map.values())

        self.rtn = Returns(settings)
        self.res = Results(settings)

        self._distros_to_compare = {'tpl': 'truncated_power_law',
                                    'exp': 'exponential',
                                    'lgn': 'lognormal'}

    # # # iteration state DEPENDENT (or aware) methods # # #

    def _log_curr_iter(self):
        # TODO: factor out repetitive log? (static: date, dynamic: group_label)
        gtyp, *date, tail = self.curr_iter_id
        grp_tail_log = (f"Analyzing {tail.name.upper()} tail of time series "
                        f"for {self.sd.grouping_type.title()} '{gtyp}' ")
        if bool(date):  # dynamic approach
            df = date[0]
            di = self.sa.get_dyn_lbd(df)
            # NOTE: di above is 1st date w/ price, not 1st date w/ return
        else:           # static approach
            di, df = self.sd.date_i, self.sd.date_f
        date_log = f"b/w [{di}, {df}]"
        print(grp_tail_log + date_log)

    @abstractmethod
    def _set_curr_input_array(self):
        # NOTE: storage posn into results_df (curr_df_pos) also set here
        pass

    def __get_xmin(self):
        rule, qnty = self.sa.xmin_rule, self.sa.xmin_qnty
        if rule in {"clauset", "manual"}:
            xmin = qnty  # ie. {None, user-input-ℝ} respectively
        elif rule == "percent":
            xmin = np.percentile(self.curr_signed_returns, qnty)
        elif rule == "std-dev":
            xmin = self.__calc_stdv_xmin(qnty)
        elif rule in {"file", "average"}:
            assert self.sa.use_dynamic,\
                ("static approach does NOT currently support passing "
                 "xmin data by file")  # TODO: add file support for -a static?
            grp, date, tail = self.curr_iter_id
            txmin = self.sa.txmin_map[tail]
            xmin = qnty.loc[date, f"{txmin} {grp}"]
            if isinstance(xmin, str) and xmin.endswith("%"):
                # b/c values containing '%' in xmins_df must be str
                percent = float(xmin[:-1])
            elif isinstance(xmin, (int, float)) and self._use_pct_file:
                if not (0 <= xmin <= 1):
                    raise TypeError("xmin percentile threshold value for "
                                    f"{self.iter_id_keys} is outside of 0-100")
                percent = xmin * 100
            else:
                pass  # numerical xmin data reaches this branch
            try:
                xmin = np.percentile(self.curr_signed_returns, percent)
            except NameError:
                xmin = float(xmin)
        else:
            raise AttributeError("this should never be reached!")
        return xmin

    def __calc_stdv_xmin(self, factor):
        mean = st.fmean(self.curr_returns_array)
        stdv = st.stdev(self.curr_returns_array)
        *_, tail = self.curr_iter_id
        assert mean < factor * stdv
        return abs(mean + tail.value * factor * stdv)  # tail.value ∈ {1, -1}

    def _fit_curr_data(self):
        data = self.curr_signed_returns
        data = data[np.nonzero(data)]  # only use non-zero elements to do Fit
        xmin = self.__get_xmin()
        self.curr_fit = Fit(data=data, xmin=xmin,
                            discrete=self.sa.fit_discretely)

    def __get_curr_rtrn_stats(self):
        # NOTE: functions in below list must match order in output_columns.yaml
        rs_fns = (len, lambda r: np.count_nonzero(r == 0), np.count_nonzero,
                  st.fmean, st.stdev, scipy.stats.skew, scipy.stats.kurtosis)
        rstats_fmap = {self.sd.rstats_collabs[i]: rs_fns[i] for i
                       in range(len(rs_fns))}
        return {rstat: rstats_fmap[rstat](self.curr_returns_array)
                for rstat in self.sd.rstats_collabs}

    def __get_curr_tail_stats(self):
        alpha, xmin, sigma = (getattr(self.curr_fit.power_law, prop)
                              for prop in ('alpha', 'xmin', 'sigma'))
        elm_in_fit = self.curr_signed_returns >= xmin
        fitted_vec = self.curr_signed_returns[elm_in_fit]
        xmax = max(fitted_vec, default=np.nan)
        xmean = fitted_vec.mean()
        xstdv = fitted_vec.std()
        abs_len = len(fitted_vec)
        if self.sa.run_ks_test is True:
            # TODO: try compute ks_pv using MATLAB engine & module, and time
            ks_pv, _ = plpva(self.curr_signed_returns, xmin, 'reps',
                             self.sa.ks_iter, 'silent')
        locs = locals()
        return {('tail-statistics', stat): locs.get(stat) for st_type, stat
                in self.sd.tstats_collabs if stat in locs}

    def __get_curr_logl_stats(self):
        # compute (R, p)-pairs (x3) using powerlaw.Fit.distribution_compare
        logl_stats = {key:
                      {stat: val for stat, val in
                       zip(('R', 'p'),
                           self.curr_fit.distribution_compare(
                               'power_law', distro,
                               normalized_ratio=True))}
                      for key, distro in self._distros_to_compare.items()}
        return {('log-likelihoods', f"{dist}_{st}"): val for dist, stats
                in logl_stats.items() for st, val in stats.items()}

    def __get_curr_plfit_stats(self):
        tail_stats = self.__get_curr_tail_stats()
        logl_stats = (self.__get_curr_logl_stats()
                      if self.sa.compare_distros else {})
        return {**tail_stats, **logl_stats}

    def _gset_curr_partial_results(self, action):
        idx, col = self.curr_df_pos  # type(idx)==str; type(col)==tuple
        tstats_map = {(col if self.sa.use_dynamic else (col,))+tuple(tsk): tsv
                      for tsk, tsv in self.__get_curr_plfit_stats().items()}

        if self.sa.calc_rtrn_stats:
            col = (col[0],) if self.sa.use_dynamic else ()
            need_rst = self.res.df.loc[idx, col + ('returns-statistics',)].hasnans
            # FIXME: NaN-check on (<grp>, 'rtrn-stats') to avoid redundant calc
            # only works for 1-proc b/c multiproc doesn't update res_df til end
            rstats_map = ({col + tuple(rsk): rsv for rsk, rsv
                           in self.__get_curr_rtrn_stats().items()}
                          if need_rst else {})
        else:
            rstats_map = {}

        # TODO: use np.ndarray instead of pd.Series (wasteful) --> order later
        curr_part_res_series = pd.Series({**tstats_map, **rstats_map})

        if action == 'store':
            self.res.df.loc[idx].update(curr_part_res_series)
            # TODO: consider using pd.DataFrame.replace(, inplace=True) instead
            # TODO: can also order stats results first, then assign to DF row
        elif action == 'return':
            return idx, curr_part_res_series
        else:
            raise AttributeError("this should never be reached!")

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
        self._gset_curr_partial_results('store')

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
        return self._gset_curr_partial_results('return')

    # runs analysis in multiprocessing mode
    def analyze_multiproc(self):
        # TODO: https://stackoverflow.com/a/52596590/5437918 (use shared DBDFs)
        iter_id_keys = tuple(self.iter_id_keys)

        # TODO: look into Queue & Pipe for sharing data
        with Pool(processes=self.sc.nproc) as pool:
            # TODO checkout .map alternatives: .imap, .map_async, etc.
            restup_ls = [restup for restup in  # TODO: optimize chunksize below
                         pool.map(self._analyze_iter, iter_id_keys)]

        # TODO: update res_df more efficiently, ex. pd.df.replace(), np.ndarray
        for restup in restup_ls:
            idx, res = restup  # if use '+' NOTE that DFs init'd w/ NaNs
            self.res.df.loc[idx].update(res)

    # top-level convenience method that autodetects how to run tail analysis
    def analyze(self):
        nproc = self.sc.nproc
        # TODO: add other conditions for analyze_sequential (ex. -a static)
        if nproc == 1:
            self.analyze_sequential()
        elif nproc > 1:
            self.analyze_multiproc()
        else:  # if 0 or negative number of processors got through to here
            raise TypeError(f'Cannot perform analysis with {nproc} processes')

    def get_resdf(self):
        # TODO: final clean ups of DF for presentation:
        #       - use .title() on all index labels, then write to file
        self.res.prettify_df()
        return self.res.df

    #  def write_results_to_file(self):
    #      self.res.prettify_df()
    #      self.res.write_df_to_file()


class StaticAnalyzer(_Analyzer):

    def __init__(self, settings):
        super().__init__(settings)
        assert not self.sa.use_dynamic
        self.iter_id_keys = product(self.sd.grouping_labs,
                                    self.sa.tails_to_anal)

    def _set_curr_input_array(self):  # TODO: pass curr_iter_id as arg???
        lab, tail = self.curr_df_pos = self.curr_iter_id
        self.curr_returns_array = self.rtn.get_returns_by_iterId(lab)
        self.curr_signed_returns = self.curr_returns_array * tail.value


class DynamicAnalyzer(_Analyzer):

    def __init__(self, settings):
        super().__init__(settings)
        assert self.sa.use_dynamic
        self.iter_id_keys = product(self.sd.grouping_labs,
                                    self.sd.anal_dates,
                                    self.sa.tails_to_anal)

    # TODO: consider vectorizing operations on all tickers
    def _set_curr_input_array(self):  # TODO: pass curr_iter_id as arg???
        sub, date, tail = self.curr_iter_id
        self.curr_df_pos = date, (sub, tail)
        self.curr_returns_array = self.rtn.get_returns_by_iterId((sub, date))
        self.curr_signed_returns = self.curr_returns_array * tail.value


# wrapper func: instantiate correct Analyzer type and run tail analysis
def analyze_tail(settings):
    Analyzer = DynamicAnalyzer if settings.anal.use_dynamic else StaticAnalyzer
    analyzer = Analyzer(settings)
    analyzer.analyze()
    analyzer.res.write_df_to_file()
    results = analyzer.get_resdf()
    print('-' * 120)
    print(results)
    print('-' * 75)
    results.info()
    print('-' * 75)
