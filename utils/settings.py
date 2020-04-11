import yaml
import pandas as pd

import enum
from types import SimpleNamespace  # TODO: consider switching to namedtuple?
from statistics import NormalDist


class Settings:

    def __init__(self, ui_options):
        for opt, val in ui_options.items():
            setattr(self, opt, val)
            # TODO: coord. w/ attrs.yaml to mark some opts private w/ _-prefix

        # core general settings (mostly data & analysis)
        self._postprocess_specific_options()
        self._gset_tail_settings()
        self._gset_dbdf_attrs()

        # domain-, functionality- & usecase- specific settings
        self._gset_grouping_info()  # must be called after _gset_dbdf_attrs()
        self._gset_avg_xmins_df()  # call after grp_info() & only -G & dynamic
        self._load_set_stats_columns_labels()
        self._gset_plot_settings()

        # instantiate the settings SimpleNamespace objects
        self._load_set_settings_config()
        self.settings = SimpleNamespace(**{ss: self._make_settings_object(ss)
                                           for ss in self._subsettings})

    # TODO: test & improve (also maybe add __repr__?)
    def __str__(self, subsett=None):  # TODO: make part of --verbose logging??
        if subsett is None:
            for ss in self._subsettings:
                print('\n' * 3 + '#' * 20)
                print(f'{ss} Settings:')
                print('#' * 20)
                for attr, val in vars(getattr(self.settings, ss)).items():
                    print('-' * 40)
                    if isinstance(val, pd.DataFrame):
                        print(f'Pandas DataFrame: {attr}')
                        print(val.info())  # TODO: use try-except on df.info?
                    else:
                        print(f'{attr}: {val}')
                print('-' * 40)
        elif subsett in self._subsettings:
            print(getattr(self.settings, subsett))
        else:
            self._validate_subsetting(subsett)
        return ''  # FIXME: instead of just print, should return proper str val

    # # methods needed for the core general settings # #

    def _postprocess_specific_options(self):
        self.approach, self._lookback, self._anal_freq = self.approach_args
        self.use_dynamic = (True if self.approach in {'rolling', 'increasing'}
                            else False)
        self.fit_discretely = True if self.data_nature == 'discrete' else False
        self.xmin_rule, self.xmin_vqty = self.xmin_args
        if self.norm_target is not None:
            self.norm_target = 'series' if self.norm_target else 'tail'
        self.run_ks_test = False if self.ks_iter <= 0 else self.run_ks_test

    def _gset_tail_settings(self):
        """Compute settings relevant to tail selection
        """
        tails_to_anal = []
        if self.anal_right:
            tails_to_anal.append(Tail.right)
        if self.anal_left:
            tails_to_anal.append(Tail.left)
        self.tails_to_anal = tuple(tails_to_anal)
        self.alpha_qntl = NormalDist().inv_cdf(1 - len(self.tails_to_anal)/2 *
                                               self.alpha_signif)

    # helper for correctly extending back DF's date-range when use_dynamic
    def __dynamize_ts_df(self, ts_df):  # ts_df: Time Series DataFrame
        assert self.use_dynamic
        full_dates = ts_df.index

        # NOTE: use (lookback - 1) b/c date analyzed incl. in lkbk window
        idx_lkbk = full_dates.get_loc(self.date_i) - (self._lookback - 1)
        assert idx_lkbk >= 0, (f"cannot lookback {self._lookback} days from "
                               f"{self.date_i} in\n{ts_df.info()}")
        lab_lkbk = full_dates[idx_lkbk]
        lab_last = self.date_f
        dynamized_df = ts_df.loc[lab_lkbk:lab_last]

        if self._anal_freq > 1:
            # slice backwards from date_i to ensure date_i is part of input
            lkbk_slice = dynamized_df.loc[self.date_i::-self._anal_freq]
            anal_slice = dynamized_df.loc[self.date_i::self._anal_freq]
            dynamized_df = pd.concat((lkbk_slice[::-1], anal_slice[1:]))

        return dynamized_df

    def _gset_dbdf_attrs(self):
        # NOTE on dbdf distinctions:
        # - full_dbdf: unfiltered DataFrame as fully loaded from input DB_FILE
        # - _tickers_dbdf: filtered by tickers (columns); has all dates (index)
        # - static_dbdf: filtered _tickers_dbdf w/ given date range to analyze
        # - dynamic_dbdf: has data going back to the earliest lookback date
        # - raw_dbdf: either dynamic_dbdf OR static_dbdf based on use_dynamic;
        #             this is the actual dbdf passed onto Analyzer

        # TODO: mark unexported options/settings w/ _-prefix? ex. _full_dates

        self._tickers_dbdf = self.full_dbdf[self.tickers]
        if self.analyze_group:
            self._partition_tickers_dbdf()

        static_dbdf = self._tickers_dbdf.loc[self.date_i: self.date_f:
                                             self._anal_freq]
        self.anal_dates = static_dbdf.index
        # TODO: use it to validate average xmin df

        if self.use_dynamic:
            dynamic_dbdf = self.__dynamize_ts_df(self._tickers_dbdf)

            # correctly set (minimum) window size based on lookback, anal_freq
            # & tau, b/c returns data pts is necssarily less than that of raw
            q, r = divmod(self._lookback, self._anal_freq)
            self.dyn_win_size = q + bool(r) - self.tau

        self.raw_dbdf = dynamic_dbdf if self.use_dynamic else static_dbdf

    # # methods relevant to group tail analysis behaviors # #

    def _partition_tickers_dbdf(self):
        if self.partition in {'country', 'maturity'}:
            # partition rules where IDs are readily parsed from ticker labels
            a, b = {'country': (0, 2), 'maturity': (3, 6)}[self.partition]
            part_ids = set(tick[a:b] for tick in self.tickers)
            part_map = {pid: [tick for tick in self.tickers if pid in tick]
                        for pid in part_ids}
        elif self.partition == 'region':
            regions_map = {'core': ('DE', 'FR', 'BE'),
                           'periphery': ('IT', 'ES', 'PT', 'IR', 'GR')}
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
        self._tickers_dbdf = pd.concat({grp: self._tickers_dbdf[tickers] for
                                        grp, tickers in part_map.items()},
                                       axis=1)  # , names=(self.partition,))
        # TODO look into pd.concat alternatives
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

    def _gset_grouping_info(self):
        self.grouping_type = GroupingName(self.partition if self.analyze_group
                                          else 'ticker')
        cix = self._tickers_dbdf.columns  # cix: column index
        self.grouping_labs = cix.levels[0] if self.analyze_group else cix

    # # methods to obtain & validate DF containing average xmins to use # #

    def __bound_and_validate_cxdf(self):
        if self.clauset_xmins_df is not None:
            bounded_cxdf = self.__dynamize_ts_df(self.clauset_xmins_df)

            # validate Dates (rows)
            assert len(bounded_cxdf) == len(self.raw_dbdf),\
                (f"Dates to be analyzed b/w [{self.date_i}, {self.date_f}] are"
                 " of DIFFERENT LENGTH for the 2 given time series data files")
            assert all(bounded_cxdf.index == self.raw_dbdf.index),\
                (f"dynamic analysis dates after looking back {self._lookback} "
                 f"days for date range [{self.date_i}, {self.date_f}] DO NOT "
                 "MATCH for the 2 given time series data files")

            # validate Group ID (columns)
            x_cols = bounded_cxdf.columns
            assert all(any(gl.lower() in xc.lower() for xc in x_cols)
                       for gl in self.grouping_labs),\
                (f"one (or more) groups [{', '.join(self.grouping_labs)}] NOT "
                 f"found in Clauset xmins file columns: '{', '.join(x_cols)}'")

            tx_map = {Tail.right: 'stp', Tail.left: 'stn'}
            assert all(any(tx_map[t] in xc.lower() for xc in x_cols)
                       for t in self.tails_to_anal),\
                ("STP and/or STN not present in Clauset xmins file for chosen "
                 f"tail(s): {', '.join([t.name for t in self.tails_to_anal])}")

            return bounded_cxdf
        return None

    def _gset_avg_xmins_df(self):
        if self.analyze_group and self.use_dynamic:

            # ASK/TODO: use truncated averging if len(xmins) < window + lag??
            # ASK: alternatively, go back to early enough date to back-calc. avg
            # ASK/CONFIRM: 0 lag means curr. date factored into average, correct?
            #              1 lag means all dates up to but not incl. curr averaged?

            window, lag = self.xmin_vqty
            bcxdf = self.__bound_and_validate_cxdf()  # bcxdf: bounded cxdf

            if bcxdf is not None:
                axdf = bcxdf.rolling(window).mean().fillna(bcxdf)
                self.avg_xmins_df = axdf.shift(lag).loc[self.date_i:]
            else:
                # TODO/FIXME: run '-x clauset', and save xmins from that to use
                raise AssertionError("Need pre-computed Clauset xmins to average")

    # # methods configuring the output results DataFrame # #

    def _load_set_stats_columns_labels(self):
        # TODO/NOTE: -G, --group dynamic cols differs slightly (xmin_today)
        cfg_bn = 'dynamic' if self.use_dynamic else 'static'
        DIR = 'config/output_columns/'  # TODO: improve package/path system
        with open(f'{DIR}/{cfg_bn}.yaml') as cfg:
            self.stats_colname, labels = tuple(
                yaml.load(cfg, Loader=yaml.SafeLoader).items())[0]

        if self.run_ks_test is False:
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

    # # methods relevant to settings needed by plotter # #

    def _gset_plot_settings(self):
        self.title_timestamp = f"Time Period: {self.date_i} - {self.date_f}"
        self.labelstep = self.__get_labelstep()
        self.returns_label = self.__get_returns_label()

    def __get_returns_label(self):
        pt_i = "P(t)"
        pt_f = f"P(t+{self.tau})"

        if self.returns_type == "raw":
            label = f"{pt_f} - {pt_i}"
        elif self.returns_type == "relative":
            label = f"{pt_f}/{pt_i} - 1.0"
        elif self.returns_type == "log":
            label = rf"$\log$({pt_f}/{pt_i})"

        if self.absolutize:
            label = f"|{label}|"

        return label

    def __get_labelstep(self):
        len_dates = len(self.anal_dates)
        _analyze_nondaily = self._anal_freq is not None and self._anal_freq > 1
        use_monthly = len_dates <= Period.ANNUAL or _analyze_nondaily
        use_quarterly = Period.ANNUAL < len_dates <= 3*Period.ANNUAL
        return (Period.MONTH if use_monthly else
                Period.QUARTER if use_quarterly else Period.BIANNUAL)

    # # methods for creating the settings SimpleNamespace object(s) # #

    def _load_set_settings_config(self):
        SETTINGS_CFG = 'config/settings.yaml'  # TODO: refactor PATH into root
        with open(SETTINGS_CFG) as cfg:
            self._settings_config = yaml.load(cfg, Loader=yaml.SafeLoader)
        self._subsettings = tuple(self._settings_config.keys())

    def _validate_subsetting(self, subsett):
        assert subsett in self._subsettings,\
            f"subsetting must be one of: {', '.join(self._subsettings)}"

    def _make_settings_object(self, subsett):
        self._validate_subsetting(subsett)
        sett_map = {}
        for sett in self._settings_config[subsett]:
            sett_map[sett] = getattr(self, sett, None)
        return SimpleNamespace(**sett_map)


class GroupingName(str):  # TODO: add allowed values constraint when have time
    """special string class used to represent the grouping type name
    """
    # FIXME: got error below when running w/ multiprocessing but not w/ single
    #        proc --> try collections.UserString? And also not w/ __new__ rm'd
    #        TypeError: __new__() missing 1 required positional argument:
    #                   'partition_choices'
    #  def __new__(cls, content, partition_choices):
    #      assert content in partition_choices,\
    #      f"GroupingStr obj must be one of: {', '.join(partition_choices)}"
    #      return str.__new__(cls, content)
    # NOTE: Assertion chk above isn't req'd, as gidx_name already constrained

    def pluralize(self):
        if self.endswith('y'):
            return self[:-1] + 'ies'
        else:
            return self + 's'


# TODO: move these Enum types into a constants.py module?

class Tail(enum.Enum):
    right = 1
    left = -1


class Period(enum.IntEnum):
    # TODO: just subclass Enum, and return PERIOD.value for self.labelstep??
    MONTH = 22
    QUARTER = 66
    BIANNUAL = 121
    ANNUAL = 252
