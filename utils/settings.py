#  import numpy as np

from types import SimpleNamespace
from statistics import NormalDist


class Settings:

    def __init__(self, ui_options):
        for opt, val in ui_options.items():
            setattr(self, opt, val)

        self._get_db_objects()
        self._set_tails()
        self._validate_xmin_args()

        self.ctrl_settings = {}
        self.data_settings = {}

    def _get_db_objects(self):
        #  self.tickers_df = self.full_dbdf[self.tickers]
        #  self.dbdf = self.full_dbdf.iloc[self.ind_i: self.ind_f + 1]
        self.dbdf = self.full_dbdf.loc[self.date_i:self.date_f, self.tickers]

        #  self.full_dates = self.full_dbdf.index
        full_dates = self.full_dbdf.index
        self.ind_i = full_dates.get_loc(self.date_i)
        self.ind_f = full_dates.get_loc(self.date_f)
        self.anal_dates = self.full_dates[self.ind_i:self.ind_f+1]

    def _set_tails(self):
        """Return relevant tail selection settings
        """
        self.use_right = True if self.tail_selection in ('right', 'both') else False
        self.use_left = True if self.tail_selection in ('left', 'both') else False

        tails_used = []
        if self.use_right:
            tails_used.append('right')
        if self.use_left:
            tails_used.append('left')
        self.tails_used = tuple(tails_used)

        mult = 0.5 if self.tail_selection == 'both' else 1
        self.alpha_qtl = NormalDist().inv_cdf(1 - mult * self.alpha_signif)

    def _validate_xmin_args(self):
        rule, *vqarg = self.xmin_args

        # confirm extra args are numeric types
        if rule != 'clauset':
            try:
                vqarg = map(float, vqarg)
            except ValueError:
                raise ValueError(f"extra arg(s) to xmin rule '{rule}' must be "
                                 f"numeric type(s), given: {', '.join(vqarg)}")

        # validate numeric args by xmin rule
        if rule == 'clauset':
            assert vqarg[0] is None,\
                ("xmin determination rule 'clauset' does not "
                 "take additional arguments")
        elif rule == 'manual':
            assert vqarg[0] == float(vqarg)  # TODO: use better check
        elif rule == 'percentile':  # ASK/TODO: use <= OR < is okay??
            assert 0 < vqarg[0] < 100,\
                ("xmin determination rule 'percentile' takes "
                 "a number between 0 and 100")
        elif rule == 'average':
            assert vqarg[0] >= vqarg[1]  # TODO: use better check

    def _get_misc_metainfo(self):

        # FIXME: should be length of spec_dates?
        self.n_vec = self.ind_f - self.ind_i + 1

        #  labelstep = (22 if n_vec <= 252 else
        #               66 if (n_vec > 252 and n_vec <= 756) else
        #               121)
        #
        #  # TODO: remove need for "spec_" variables
        #  if anal_freq > 1:
        #      spec_dates = dates[::anal_freq]
        #      spec_labelstep = 22
        #  elif anal_freq == 1:
        #      spec_dates = dates
        #      spec_labelstep = labelstep
        #  n_spdt = len(spec_dates)

    # TODO: OR distinguish b/w analysis vs. control-flow settings!!
    def set_ctrl_settings(self):
        pass

    def set_data_settings(self):
        pass
