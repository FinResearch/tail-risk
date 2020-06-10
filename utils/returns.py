import numpy as np

from abc import ABC, abstractmethod


class Returns:

    def __init__(self, settings):
        self.sd = settings.data
        self.sr = settings.rtrn
        self.sa = settings.anal

        if self.sa.approach == 'static':
            Normalizer = StaticNormalizer
        elif self.sa.approach == 'monthly':
            Normalizer = MonthlyNormalizer
        else:
            Normalizer = DynamicNormalizer

        returns_df = self.__compute_returns_df()
        self.normalizer = Normalizer(settings, returns_df)

    def __compute_returns_df(self):
        tau = self.sr.tau

        nan_pad = np.full(tau, np.nan)
        print(f"{self.sr.returns_type.upper()} returns selected")
        # TODO: move above printing into verbosity logging

        def get_returns(series):  # calculates returns for a series
            pt_i = series[:-tau]
            pt_f = series[tau:]
            if self.sr.returns_type == "raw":
                returns = pt_f - pt_i
            elif self.sr.returns_type == "relative":
                returns = pt_f / pt_i - 1.0
            elif self.sr.returns_type == "log":
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
    def get_returns_by_iterId(self, iterId):
        return self.normalizer.get_returns_array(iterId)


class _Normalizer(ABC):
    """NOTE: in all cases, if neither standardize nor absolutize is True, then
    the returns data would simply pass through the Normalizer object unmodified
    """

    def __init__(self, settings, returns_df):
        # TODO: take input vector directly from data settings?
        self.sd = settings.data
        self.sr = settings.rtrn
        self.sa = settings.anal
        self.returns_df = returns_df

    @abstractmethod
    def _get_returns_PdObj(self, iterId):
        pass

    def __normalize_numpy(self, X):  # X must be a numpy.ndarray
        if self.sr.standardize:
            X = (X - X.mean()) / X.std(ddof=1)
        if self.sr.absolutize:
            X = np.abs(X)
        return X

    def get_returns_array(self, iterId):
        # TODO: implement std/abs for when target is 'tail' in individual mode

        # TODO optimize below: no need to .flatten() when not analyze_group??
        rtrn = self._get_returns_PdObj(iterId).to_numpy().flatten()
        rtrn = rtrn[~np.isnan(rtrn)]

        # normalize after grouping in -G mode
        if self.sr.norm_after:
            # TODO/ASK: take care of when to absolutize for -G, since --norm_after is the default
            # ASK/TODO: do --std & --abs necessarily have the same timing when both are specified???
            rtrn = self.__normalize_numpy(rtrn)
        return rtrn


class StaticNormalizer(_Normalizer):

    def __init__(self, settings, returns_df):
        super().__init__(settings, returns_df)
        assert not self.sa.use_dynamic

        if self.sr.standardize:  # TODO: calc always if getting moments here
            # means & stds should be Pandas Series (1 per ticker)
            self.means = self.returns_df.mean()  # NOTE: NaNs ignored by dflt
            self.stds = (self.returns_df.std()   # NOTE: NaNs ignored by dflt
                         if len(self.returns_df) > 1 else None)
            self.stdzd_cols_df = (self.returns_df - self.means) / self.stds

    # NOTE: for non-G, std/abs only applies to static when target is 'series'
    def _get_returns_PdObj(self, iterId):
        group = iterId
        rtrn = self.returns_df[group]
        if self.sr.standardize:
            rtrn = self.stdzd_cols_df[group]
        if self.sr.absolutize:
            rtrn = np.abs(rtrn)
        return rtrn

    # FIXME/TODO: implement std/abs when target is 'tail' in individual mode


# TODO: consider making MonthlyNormalizer a subclass of DynamicNormalizer
class MonthlyNormalizer(_Normalizer):

    def __init__(self, settings, returns_df):
        super().__init__(settings, returns_df)
        assert self.sa.approach == 'monthly'

        if self.sr.standardize:  # TODO: calc always if getting moments here
            # means & stds should be Pandas DFs (1 per date & per ticker)
            # initialize 2 DFs for storing the means & std-devs
            self.means = self.returns_df.loc[self.sd.anal_dates]
            self.stds = self.returns_df.loc[self.sd.anal_dates]
            for date in self.sd.anal_dates:
                _, d_i, d_f = self.sr.monthly_bounds[date[3:]]
                # NOTE: by default, DF mean() & std() ignore NaNs & use axis 0
                self.means.loc[date] = self.returns_df.loc[d_i:d_f].mean()
                self.stds.loc[date] = self.returns_df.loc[d_i:d_f].std()

    def _get_returns_PdObj(self, iterId):
        group, date = iterId
        lkbd = self.sr.monthly_bounds[date[3:]][1]
        rtrn = self.returns_df.loc[lkbd:date, group]
        if self.sr.standardize:
            stat_loc = (date, group)
            rtrn = (rtrn - self.means.loc[stat_loc]) / self.stds.loc[stat_loc]
        if self.sr.absolutize:
            rtrn = np.abs(rtrn)
        return rtrn


class DynamicNormalizer(_Normalizer):

    def __init__(self, settings, returns_df):
        super().__init__(settings, returns_df)
        assert self.sa.use_dynamic and self.sa.approach != 'monthly'

        self.dates = self.returns_df.index

        self._win_type = {'rolling': 'rolling', 'increasing': 'expanding'}.\
            get(self.sa.approach)
        self.rtrn_window = getattr(self.returns_df,
                                   self._win_type)(self.sr.dyn_win_size)

        if self.sr.standardize:  # TODO: calc always if getting moments here
            # means & stds should be Pandas DFs (1 per date & per ticker)
            self.means = self.__get_window_stat('mean')
            self.stds = (self.__get_window_stat('std')
                         if len(self.returns_df) > 1 else None)

    def __get_window_stat(self, stat):
        window_stat = getattr(self.rtrn_window, stat)().dropna()
        assert len(window_stat) == len(self.sd.anal_dates),\
            (f'cannot calculate suitable {stat.upper()}s for analysis dates '
             f"[{', '.join(self.sd.anal_dates)}] from {self._win_type} "
             f'window of minimum size {self.sr.dyn_win_size}, constructed '
             f'from DataFrame below:\n\n{self.returns_df}\n')
        return window_stat

    def __get_lookback_label(self, date):
        lkb = (self.dates.get_loc(date) - self.sr.dyn_win_size + 1
               if self.sa.approach == 'rolling' else 0)
        assert lkb >= 0  # this necessarily must be True, otherwise bug
        return self.dates[lkb]

    def _get_returns_PdObj(self, iterId):
        group, date = iterId
        lkbd = self.__get_lookback_label(date)  # lookback date
        rtrn = self.returns_df.loc[lkbd:date, group]
        if self.sr.standardize:
            stat_loc = (date, group)
            rtrn = (rtrn - self.means.loc[stat_loc]) / self.stds.loc[stat_loc]
        if self.sr.absolutize:
            rtrn = np.abs(rtrn)
        return rtrn


class ReturnsIter:

    def __init__(self, configurer, identifier):
        self.cfg = configurer
        self.id = identifier

        # instantiate a new Returns obj for each analysis iteration
        # make returns_array, mean, std-dev, skew, kurt, n_returns into attrs
