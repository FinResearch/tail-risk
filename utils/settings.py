#  import numpy as np

from types import SimpleNamespace
from statistics import NormalDist


class Settings:

    def __init__(self, ui_options):
        for opt, val in ui_options.items():
            setattr(self, opt, val)

        # get (extract) & set (compute) specific settings from given opts
        self._get_anal_freq()
        self._get_db_objects()
        self._get_xmin_args()
        self._set_tails()
        self._set_labelstep()

        self.ctrl_settings = {}
        self.data_settings = {}

    def _get_anal_freq(self):
        self.approach, self.anal_freq = self.approach_args

    def _get_db_objects(self):
        #  self.tickers_df = self.full_dbdf[self.tickers]
        #  self.dbdf = self.full_dbdf.iloc[self.ind_i: self.ind_f + 1]
        self.dbdf = self.full_dbdf.loc[self.date_i:self.date_f, self.tickers]

        #  self.full_dates = self.full_dbdf.index
        full_dates = self.full_dbdf.index
        self.ind_i = full_dates.get_loc(self.date_i)  # TODO:inds still needed?
        self.ind_f = full_dates.get_loc(self.date_f)
        self.anal_dates = self.full_dates[self.ind_i:
                                          self.ind_f+1:
                                          self.anal_freq]

    def _get_xmin_args(self):
        # TODO: it's possible to compute all xmin_rules except for 'average'
        # for 'percentile', just apply to 'data' knowing the 'lookback', etc.
        self.xmin_rule, self.xmin_vqty = self.xmin_args

    def _set_tails(self):
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
        self.alpha_qntl = NormalDist().inv_cdf(1 - mult * self.alpha_signif)

    def _set_labelstep(self):
        # FIXME: should be length of spec_dates?
        n_vec = self.ind_f - self.ind_i + 1
        labelstep = (22 if n_vec <= 252 else
                     66 if 252 < n_vec <= 756 else 121)
        self.labelstep = 22 if self.anal_freq > 1 else labelstep

    # TODO: OR distinguish b/w analysis vs. control-flow settings!!
    def set_ctrl_settings(self):
        pass

    def set_data_settings(self):
        pass
