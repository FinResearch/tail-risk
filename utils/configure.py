import numpy as np

from abc import ABC, abstractmethod


class DataConfigurer:

    def __init__(self, settings):
        self.sd = settings.data
        self.sa = settings.anal

        if self.sa.use_dynamic:
            #  lkb = self.sa.lookback
            #  idx_i = self.sd.date_i_idx - lkb + 1
            idx_i = self.sd.date_i_idx - self.sa.lookback + 1
            idx_f = self.sd.full_dates.get_loc(self.sd.date_f)
            # TODO: add the correctly time-range-sliced dbdf into settings.py?
            self.raw_dbdf = self.sd.dynamic_dbdf.iloc[idx_i:idx_f+1]
            # FIXME/TODO: the above need to be sliced by anal_freq
        else:
            self.raw_dbdf = self.sd.static_dbdf
        returns_df = self.__compute_returns_df()

        Normalizer = (DynamicNormalizer if self.sa.use_dynamic else
                      StaticNormalizer)
        self.normalizer = Normalizer(settings, returns_df)

    def __compute_returns_df(self):
        nan_pad = np.full(self.sa.tau, np.nan)

        # TODO: shove below printing into verbosity logging
        print(f"{self.sa.returns_type.upper()} returns selected")

        def get_returns(series):  # calculates returns for a series
            pt_i = series[:-self.sa.tau]
            pt_f = series[self.sa.tau:]
            if self.sa.returns_type == "raw":
                returns = pt_f - pt_i
            elif self.sa.returns_type == "relative":
                returns = pt_f / pt_i - 1.0
            elif self.sa.returns_type == "log":
                returns = np.log(pt_f/pt_i)
            return np.hstack((nan_pad, returns))

        returns_df = self.raw_dbdf.apply(get_returns, raw=True).dropna()
        assert len(returns_df) == len(self.raw_dbdf) - self.sa.tau
        #  returns_df = self.raw_dbdf.apply(get_returns, raw=True)

        #  # construct a new column repr dates used to calc each return
        #  tau = self.sa.tau
        #  idx0 = self.raw_dbdf.index
        #  irng = range(tau, idx0.get_loc(self.sd.date_f) + 1)
        #  idx1 = (f"{idx0[i-tau]} ⟶  {idx0[i]}" for i in irng)
        #  assert len(returns_df) == len(idx1)
        #  returns_df['returns_date_range'] = idx1

        return returns_df

    # the public method to be called by an Analyzer instance
    def get_data(self, iter_id):
        return self.normalizer.get_normed_data(iter_id)


class _Normalizer(ABC):
    """NOTE: in all cases, if neither standardize nor absolutize is True, then
    the returns data would simply pass through the Normalizer object unmodified
    """

    def __init__(self, settings, returns_df):
        # TODO: take input vector directly from data settings?
        self.sd = settings.data
        self.sa = settings.anal
        self.returns_df = returns_df

    @abstractmethod
    def _norm_tickers(self, iter_id):
        pass

    def __normalize_numpy(self, X):  # X must be a numpy.ndarray
        if self.sa.standardize:
            X = (X - X.mean()) / X.std(ddof=1)
        if self.sa.absolutize:
            X = np.abs(X)
        return X

    def get_normed_data(self, iter_id):
        # TODO: implement std/abs for when target is 'tail' in individual mode

        # TODO optimize below: no need to .flatten() when not analyze_group??
        data = self._norm_tickers(iter_id).to_numpy().flatten()

        # normalize after grouping in -G mode
        if self.sa.norm_after:
            data = self.__normalize_numpy(data)
        return data


class StaticNormalizer(_Normalizer):

    def __init__(self, settings, returns_df):
        super().__init__(settings, returns_df)
        assert not self.sa.use_dynamic

        # means & stds should be Pandas Series here
        self.means = self.returns_df.mean()
        self.stds = self.returns_df.std()

        self.stdzd_cols_df = (self.returns_df - self.means) / self.stds

    # NOTE: for non-G, std/abs only applies to static when target is 'series'
    def _norm_tickers(self, iter_id):
        group, _ = iter_id
        data = self.returns_df[group]
        if self.sa.standardize:
            data = self.stdzd_cols_df[group]
        if self.sa.absolutize:
            data = np.abs(data)
        return data
    # TODO: implement std/abs for when target is 'tail' in individual mode


class DynamicNormalizer(_Normalizer):

    def __init__(self, settings, returns_df):
        super().__init__(settings, returns_df)
        assert self.sa.use_dynamic

        # NOTE: blow offset req'd b/c returns necssarily has less data than raw
        self.lb_off = self.sa.lookback - self.sa.tau  # lookback offset

        win_type = {'rolling': 'rolling', 'increasing': 'expanding'}.\
            get(self.sa.approach)
        self.data_window = getattr(self.returns_df, win_type)(self.lb_off)

        # means & stds should be Pandas DataFrame here
        self.means = self.__get_window_stat('mean')
        self.stds = self.__get_window_stat('std')

    def __get_window_stat(self, stat):
        assert hasattr(self, 'data_window')
        window_stat = getattr(self.data_window, stat)().dropna()
        assert len(window_stat) == len(self.sd.anal_dates)
        return window_stat

    def __get_lookback_label(self, date):
        dates = self.returns_df.index
        lkb = (dates.get_loc(date) - self.lb_off + 1
               if self.sa.approach == 'rolling' else 0)
        assert lkb >= 0
        return dates[lkb]

    def _norm_tickers(self, iter_id):
        group, (_, date), _ = iter_id
        lkbd = self.__get_lookback_label(date)  # lookback date

        data = self.returns_df.loc[lkbd:date, group]

        if self.sa.standardize:
            stat_loc = (date, group)
            data = (data - self.means.loc[stat_loc]) / self.stds.loc[stat_loc]
        if self.sa.absolutize:
            data = np.abs(data)

        return data
