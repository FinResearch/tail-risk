import yaml

from enum import IntEnum
from types import SimpleNamespace
from statistics import NormalDist


class Settings:

    def __init__(self, ui_options):
        for opt, val in ui_options.items():
            setattr(self, opt, val)

        # get (extract) & set (compute) specific settings from given opts
        self._set_approach_analfreq()
        self._set_xmin_args()
        self._gset_tail_attrs()
        self._gset_dbdf_attrs()
        self._gset_labelstep()  # TODO: call under another method??

        # instantiate the settings SimpleNamespace objects
        self.settings_config = self._load_settings_config()
        self.ctrl_settings = self._make_settings_object('ctrl')
        self.data_settings = self._make_settings_object('data')

    def _set_approach_analfreq(self):
        self.approach, self.anal_freq = self.approach_args

    def _set_xmin_args(self):
        # TODO: should be possible to compute all xmin_rules except 'average';
        # ex. for 'percentile', as 'lookback' is known, just compute on dbdf
        self.xmin_rule, self.xmin_vqty = self.xmin_args

    def _gset_tail_attrs(self):
        """Return relevant tail selection settings
        """
        self.use_right = True if self.tail_selection in ('right', 'both') else False
        self.use_left = True if self.tail_selection in ('left', 'both') else False

        tails_to_use = []
        if self.use_right:
            tails_to_use.append('right')
        if self.use_left:
            tails_to_use.append('left')
        self.tails_to_use = tuple(tails_to_use)

        mult = 0.5 if self.tail_selection == 'both' else 1
        self.alpha_qntl = NormalDist().inv_cdf(1 - mult * self.alpha_significance)

    def _set_labelstep(self):
        # CONFIRM/FIXME: should be length of spec_dates?
        #  n_vec = self.ind_f - self.ind_i + 1
        #  labelstep = (22 if n_vec <= 252 else
        #               66 if 252 < n_vec <= 756 else 121)
        labelstep = (22 if self.len_anal_dates <= 252 else
                     66 if 252 < self.len_anal_dates <= 756 else 121)
        self.labelstep = 22 if self.anal_freq > 1 else labelstep
    def _gset_dbdf_attrs(self):
        # NOTE on dbdf distinctions:
        # - full_dbdf: unfiltered DataFrame as fully loaded from input DB_FILE
        # - dynamic_dbdf: filtered by tickers (columns); has all dates (index)
        # - static_dbdf: filtered above by given range of dates to analyze

        self.full_dates = self.full_dbdf.index
        self.date_i_idx = self.full_dates.get_loc(self.date_i)

        self.dynamic_dbdf = self.full_dbdf[self.tickers]
        if self.analyze_group:
            self.__partition_dbdf()
            print(self.part_map)

        self.static_dbdf = self.dynamic_dbdf.loc[self.date_i: self.date_f]
        self.anal_dates = self.static_dbdf.index[::self.anal_freq]
        self.len_dates = len(self.anal_dates)

    def _gset_labelstep(self):
        _analyze_nondaily = self.anal_freq is not None and self.anal_freq > 1
        use_monthly = self.len_dates <= Period.ANNUAL or _analyze_nondaily
        use_quarterly = Period.ANNUAL < self.len_dates <= 3*Period.ANNUAL

        self.labelstep = (Period.MONTH if use_monthly else
                          Period.QUARTER if use_quarterly else
                          Period.BIANNUAL)

    # # methods for creating the settings SimpleNamespace object(s) # #

    def _load_settings_config(self):
        SETTINGS_CFG = 'config/settings.yaml'  # TODO: refactor PATH into root
        with open(SETTINGS_CFG) as cfg:
            return yaml.load(cfg, Loader=yaml.SafeLoader)

    def _valid_settings_cls(self, sett_cls):
        sett_classes = self.settings_config.keys()
        assert sett_cls in sett_classes,\
            f"settings class name must be one of: {', '.join(sett_classes)}"

    def _make_settings_object(self, sett_cls):
        self._valid_settings_cls(sett_cls)
        sett_map = {}
        for sett in self.settings_config[sett_cls]:
            sett_map[sett] = getattr(self, sett, None)
        return SimpleNamespace(**sett_map)

    # TODO: remove this method, and just pass entire settings object?
    def get_settings_object(self, sett_cls):
        self._valid_settings_cls(sett_cls)
        return getattr(self, f'{sett_cls}_settings')


class Period(IntEnum):
    MONTH = 22
    QUARTER = 66
    BIANNUAL = 121
    ANNUAL = 252
