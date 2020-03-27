import yaml
import pandas as pd

from enum import IntEnum
from types import SimpleNamespace  # TODO: consider switching to namedtuple?
from statistics import NormalDist


class Settings:

    def __init__(self, ui_options):
        for opt, val in ui_options.items():
            setattr(self, opt, val)

        # get (compute/calc) & set (extract) specific settings from given opts
        self._postprocess_specific_options()
        self._gset_tail_settings()
        self._gset_dbdf_attrs()
        self._gset_labelstep()  # TODO:call in other mtd w/ anal_freq & len_dts
        self._gset_grouping_info()  # must be called after _gset_dbdf_attrs()
        self._load_set_stats_columns_labels()

        # instantiate the settings SimpleNamespace objects
        self._load_set_settings_config()
        self.settings = SimpleNamespace(**{ss: self._make_settings_object(ss)
                                           for ss in ('ctrl', 'data', 'anal')})

    # TODO: test & improve (also maybe add __str__?)
    def __repr__(self, sub_sett=None):
        sub_settings = {'ctrl', 'data', 'anal'}
        if sub_sett is None:
            print(self)
        elif sub_sett in sub_settings:
            print(getattr(self, sub_sett))
        else:
            raise AttributeError(
                f"sub-setting must be one of {', '.join(sub_settings)}"
                f"given: '{sub_sett}'"
            )

    def _postprocess_specific_options(self):
        self.approach, self.anal_freq = self.approach_args
        self.use_static = True if self.approach == 'static' else False
        self.xmin_rule, self.xmin_vqty = self.xmin_args
        self.ks_flag = False if self.ks_iter <= 0 else self.ks_flag

    # TODO: should be possible to get all xmin_val for all xmin_rule except
    # 'average'; for percentile, use lookback to slice all input arrays
    def __compute_percentile_xmin(self):
        pass

    def _gset_tail_settings(self):
        """Compute settings relevant to tail selection
        """
        use_right = True if self.tail_selection in {'right', 'both'} else False
        use_left = True if self.tail_selection in {'left', 'both'} else False

        tails_to_use = []
        if use_right:
            tails_to_use.append('right')
        if use_left:
            tails_to_use.append('left')
        self.tails_to_use = tuple(tails_to_use)
        self.n_tails = len(self.tails_to_use)

        mult = 0.5 if self.tail_selection == 'both' else 1
        self.alpha_qntl = NormalDist().inv_cdf(1 - mult * self.alpha_signif)

    def _gset_dbdf_attrs(self):
        # NOTE on dbdf distinctions:
        # - full_dbdf: unfiltered DataFrame as fully loaded from input DB_FILE
        # - dynamic_dbdf: filtered by tickers (columns); has all dates (index)
        # - static_dbdf: filtered above by given range of dates to analyze

        self.full_dates = self.full_dbdf.index
        self.date_i_idx = self.full_dates.get_loc(self.date_i)

        self.dynamic_dbdf = self.full_dbdf[self.tickers]
        if self.analyze_group:
            self._partition_dynamic_dbdf()

        self.static_dbdf = self.dynamic_dbdf.loc[self.date_i: self.date_f]
        self.anal_dates = self.static_dbdf.index[::self.anal_freq]
        self.len_dates = len(self.anal_dates)  # used by _gset_labelstep

    def _gset_labelstep(self):
        _analyze_nondaily = self.anal_freq is not None and self.anal_freq > 1
        use_monthly = self.len_dates <= Period.ANNUAL or _analyze_nondaily
        use_quarterly = Period.ANNUAL < self.len_dates <= 3*Period.ANNUAL

        self.labelstep = (Period.MONTH if use_monthly else
                          Period.QUARTER if use_quarterly else
                          Period.BIANNUAL)

    def _partition_dynamic_dbdf(self):
        if self.partition in {'country', 'maturity'}:
            # partition rules where IDs are readily parsed from ticker labels
            a, b = {'country': (0, 2), 'maturity': (3, 6)}[self.partition]
            part_ids = set(tick[a:b] for tick in self.tickers)
            part_map = {pid: [tick for tick in self.tickers if pid in tick]
                        for pid in part_ids}
        elif self.partition == 'region':
            regions_map = {'Core': ('DE', 'FR', 'BE'),
                           'Periphery': ('IT', 'ES', 'PT', 'IR', 'GR')}
            part_map = {region: ticks for region, ticks in
                        {region: [tick for tick in self.tickers if
                                  any(cid in tick for cid in countries)]
                         for region, countries in regions_map.items()}.items()
                        if ticks}  # the outter dict-comp filters out [] region
            #  if self.partition_group_leftovers:  # TODO: opt not yet usable
            #  part_map['leftovers'] = [tick for tick in self.tickers if
            #                                all(tick not in group for group
            #                                    in part_map.values())]

        # set partition groups as the top-level column label
        # TODO: use slice+loc to org ticks then set toplvl-idx to rm pd depend
        self.dynamic_dbdf = pd.concat({grp: self.dynamic_dbdf[tickers] for
                                       grp, tickers in part_map.items()},
                                      axis=1)  # , names=(self.partition,))
        # TODO look into pd.concat alternatives
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

    def _gset_grouping_info(self):
        # FIXME: doesn't work for --partition=region --> KeyError: Periphery
        self.grouping_type = self.partition if self.analyze_group else 'ticker'
        cix = self.dynamic_dbdf.columns  # cix: column index
        self.grouping_labs = cix.levels[0] if self.analyze_group else cix

    def _load_set_stats_columns_labels(self):
        # TODO/NOTE: -G, --group dynamic cols differs slightly (xmin_today)
        cfg_bn = 'static' if self.use_static else 'dynamic'
        DIR = 'config/output_columns/'  # TODO: improve package/path system
        with open(f'{DIR}/{cfg_bn}.yaml') as cfg:
            self.stats_colname, labels = tuple(
                yaml.load(cfg, Loader=yaml.SafeLoader).items())[0]

        if self.ks_flag is False:
            labels.remove('ks_pv')

        self.stats_collabs = [(lab, '') if isinstance(lab, str) else lab for
                              lab in self.__make_logl_column_labels(labels)]

    # helper func called in _load_set_stats_columns_labels
    def __make_logl_column_labels(self, labels):
        ll_labs = [(i, lab) for i, lab in enumerate(labels)
                   if lab.startswith('ll_')]
        for i, lab in reversed(ll_labs):
            # insert and pop in reverse order to preserve validity of index i
            labels.insert(i + 1, (lab, 'p'))
            labels.insert(i + 1, (lab, 'R'))
            labels.pop(i)
        return labels

    # # methods for creating the settings SimpleNamespace object(s) # #

    def _load_set_settings_config(self):
        SETTINGS_CFG = 'config/settings.yaml'  # TODO: refactor PATH into root
        with open(SETTINGS_CFG) as cfg:
            self.settings_config = yaml.load(cfg, Loader=yaml.SafeLoader)

    def _valid_settings_cls(self, sub_sett):
        sub_settings = set(self.settings_config.keys())
        # FIXME: find more potential membership testing cases like the below
        assert sub_sett in sub_settings,\
            f"settings class name must be one of: {', '.join(sub_settings)}"

    def _make_settings_object(self, sub_sett):
        self._valid_settings_cls(sub_sett)
        sett_map = {}
        for sett in self.settings_config[sub_sett]:
            sett_map[sett] = getattr(self, sett, None)
        return SimpleNamespace(**sett_map)


class Period(IntEnum):
    MONTH = 22
    QUARTER = 66
    BIANNUAL = 121
    ANNUAL = 252
