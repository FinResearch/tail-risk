import numpy as np

from abc import ABC, abstractmethod


class Returns:

    def __init__(self, settings):
        self.sd = settings.data
        self.sa = settings.anal

        returns_df = self.__compute_returns_df()
        Normalizer = (DynamicNormalizer if self.sa.use_dynamic else
                      StaticNormalizer)
        self.normalizer = Normalizer(settings, returns_df)

    def __compute_returns_df(self):
        tau = self.sa.tau

        nan_pad = np.full(tau, np.nan)
        print(f"{self.sa.returns_type.upper()} returns selected")
        # TODO: move above printing into verbosity logging

        def get_returns(series):  # calculates returns for a series
            pt_i = series[:-tau]
            pt_f = series[tau:]
            if self.sa.returns_type == "raw":
                returns = pt_f - pt_i
            elif self.sa.returns_type == "relative":
                returns = pt_f / pt_i - 1.0
            elif self.sa.returns_type == "log":
                returns = np.log(pt_f/pt_i)
            return np.hstack((nan_pad, returns))

        p_len = len(self.sd.price_dbdf)
        assert p_len > tau,\
            ('cannot calculate returns from time series price data of length '
             f'{p_len} using tau/delta of {tau} days')

        returns_df = self.sd.price_dbdf.apply(get_returns, raw=True).iloc[tau:]

        # TODO: can add below info as DEBUG logging
        #  # construct a new column repr dates used to calc each return
        #  idx0 = self.sd.price_dbdf.index
        #  irng = range(tau, idx0.get_loc(idx0[-1]) + 1)
        #  idx1 = [f"{idx0[i-tau]} ⟶  {idx0[i]}" for i in irng]
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
        data = data[~np.isnan(data)]

        # normalize after grouping in -G mode
        if self.sa.norm_after:
            # TODO/ASK: take care of when to absolutize for -G, since --norm_after is the default
            # ASK/TODO: do --std & --abs necessarily have the same timing when both are specified???
            data = self.__normalize_numpy(data)
        return data


class StaticNormalizer(_Normalizer):

    def __init__(self, settings, returns_df):
        super().__init__(settings, returns_df)
        assert not self.sa.use_dynamic

        if self.sa.standardize:  # TODO: calc always if getting moments here
            # means & stds should be Pandas Series here (1 moment per ticker)
            self.means = self.returns_df.mean()  # NOTE: NaNs ignored by dflt
            self.stds = (self.returns_df.std()   # NOTE: NaNs ignored by dflt
                         if len(self.returns_df) > 1 else None)
            self.stdzd_cols_df = (self.returns_df - self.means) / self.stds

    # NOTE: for non-G, std/abs only applies to static when target is 'series'
    def _norm_tickers(self, norm_id):
        group = norm_id
        data = self.returns_df[group]
        if self.sa.standardize:
            data = self.stdzd_cols_df[group]
        if self.sa.absolutize:
            data = np.abs(data)
        return data

    # FIXME/TODO: implement std/abs when target is 'tail' in individual mode


class DynamicNormalizer(_Normalizer):

    def __init__(self, settings, returns_df):
        super().__init__(settings, returns_df)
        assert self.sa.use_dynamic

        self.dates = self.returns_df.index

        self._win_type = {'rolling': 'rolling', 'increasing': 'expanding'}.\
            get(self.sa.approach)
        self.data_window = getattr(self.returns_df,
                                   self._win_type)(self.sa.dyn_win_size)

        if self.sa.standardize:  # TODO: calc always if getting moments here
            # means & stds should be Pandas DataFrame here (1 moment per date)
            self.means = self.__get_window_stat('mean')
            self.stds = (self.__get_window_stat('std')
                         if len(self.returns_df) > 1 else None)

    def __get_window_stat(self, stat):
        window_stat = getattr(self.data_window, stat)().dropna()
        assert len(window_stat) == len(self.sd.anal_dates),\
            (f'cannot calculate suitable {stat.upper()}s for analysis dates '
             f"[{', '.join(self.sd.anal_dates)}] from {self._win_type} "
             f'window of minimum size {self.sa.dyn_win_size}, constructed '
             f'from DataFrame below:\n\n{self.returns_df}\n')
        return window_stat

    def __get_lookback_label(self, date):
        lkb = (self.dates.get_loc(date) - self.sa.dyn_win_size + 1
               if self.sa.approach == 'rolling' else 0)
        assert lkb >= 0  # this necessarily must be True, otherwise bug
        return self.dates[lkb]

    def _norm_tickers(self, norm_id):
        group, date = norm_id
        lkbd = self.__get_lookback_label(date)  # lookback date

        data = self.returns_df.loc[lkbd:date, group]

        if self.sa.standardize:
            stat_loc = (date, group)
            data = (data - self.means.loc[stat_loc]) / self.stds.loc[stat_loc]
        if self.sa.absolutize:
            data = np.abs(data)

        return data


class ReturnsIter:

    def __init__(self, configurer, identifier):
        self.cfg = configurer
        self.id = identifier

        # instantiate a new Returns obj for each analysis iteration
        # make returns_array, mean, std-dev, skew, kurt, n_returns into attrs
