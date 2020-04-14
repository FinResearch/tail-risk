import yaml
import pandas as pd

import enum
from types import SimpleNamespace  # TODO: consider switching to namedtuple?
from statistics import NormalDist


class Settings:

    def __init__(self, user_inputs):
        self._user_inputs = user_inputs
        for opt, val in self._user_inputs.items():
            setattr(self, opt, val)
            # TODO: coord. w/ attrs.yaml to mark some opts private w/ _-prefix

        # core general settings (mostly data & analysis)
        self._postprocess_specific_options()
        self._gset_tail_settings()
        self._gset_dbdf_attrs()

        # domain-, functionality- & usecase- specific settings
        self._gset_grouping_info()  # must be called after _gset_dbdf_attrs()
        self._load_set_stats_columns_labels()
        if self.plot_results:
            self._gset_plot_settings()
        if isinstance(self.xmin_qnty, pd.DataFrame):  # only w/ {file, average}
            # validation must be after _gset_grouping_info()
            self._validate_xmins_df_statcols()

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

        self.xmin_rule, self.xmin_qnty = self.xmin_args
        self.tst_map = {Tail.right: 'STP', Tail.left: 'STN'}  # used rule: file
        if self.xmin_rule == "average":
            self._gset_xmins_df_for_average()
            # TODO: consider performing averaging of xmins from passed file
            # --> see original implementation in commit f12b505315 & prior

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

    def __get_back_date_label(self, n_back, date0=None, dates_ix=None):
        dates_ix = self.full_dbdf.index if dates_ix is None else dates_ix
        date0 = self.date_i if date0 is None else date0
        # NOTE: use (n_back - 1) b/c date0 incl. in back window by default
        idx_back = dates_ix.get_loc(date0) - (n_back - 1)
        if idx_back < 0:  # when n_back needed is beyond the given data
            raise ValueError(f"cannot go back {n_back} full days from "
                             f"{date0} in DateIndex:\n\n{dates_ix}\n")
        return dates_ix[idx_back]

    # helper for correctly extending back DF's date-range when use_dynamic
    def __dynamize_ts_df(self, ts_df, back_date, end_date=None):
        """accepts a DataFrame to slice, and an argument that can be either:
        i)  an integer num of dates to go back from self.date_i (inclusive)
        ii) a string representing the actual date label to go back to
        """
        assert self.use_dynamic
        if isinstance(back_date, int):
            back_date = self.__get_back_date_label(back_date)
        end_date = self.date_f if end_date is None else end_date
        dynamized_df = ts_df.loc[back_date:end_date]
        if self._anal_freq > 1:
            # slice backwards from date_i to ensure date_i is part of final DF
            back_slice = dynamized_df.loc[self.date_i::-self._anal_freq]
            # TODO/FIXME: self.date_i pins final DF -> could be problematic
            # when pre-computing Clauset xmins if anal_freq ≠ 1
            anal_slice = dynamized_df.loc[self.date_i::self._anal_freq]
            dynamized_df = pd.concat((back_slice[::-1], anal_slice[1:]))
        return dynamized_df

    def _gset_dbdf_attrs(self):
        # Note on dbdf distinctions:
        # - full_dbdf: unfiltered DataFrame as fully loaded from input DB_FILE
        # - _tickers_dbdf: filtered by tickers (columns); has all dates (index)
        # - static_dbdf: filtered _tickers_dbdf w/ given date range to analyze
        # - dynamic_dbdf: has data going back to the earliest lookback date
        # - raw_dbdf: either dynamic_dbdf OR static_dbdf based on use_dynamic;
        #             this is the actual dbdf passed onto Analyzer

        self._tickers_dbdf = self.full_dbdf[self.tickers]
        if self.analyze_group:
            self._partition_tickers_dbdf()

        static_dbdf = self._tickers_dbdf.loc[self.date_i: self.date_f:
                                             self._anal_freq]
        self.anal_dates = static_dbdf.index
        # TODO: use it to validate average xmin df

        if self.use_dynamic:
            dynamic_dbdf = self.__dynamize_ts_df(self._tickers_dbdf,
                                                 self._lookback)
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
                              lab in self.__structure_column_labels(labels)]

    # helper func called in _load_set_stats_columns_labels
    def __structure_column_labels(self, labels):
        if self.compare_distros:
            ll_labs = [(i, lab) for i, lab in enumerate(labels)
                       if lab.startswith('ll_')]
            for i, lab in reversed(ll_labs):
                # insert & pop in reverse order to preserve validity of idx, i
                labels.insert(i + 1, (lab, 'p'))
                labels.insert(i + 1, (lab, 'R'))
                labels.pop(i)
        else:
            labels = [lab for lab in labels if not lab.startswith('ll_')]
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

    # # methods to obtain & validate DF containing xmins to directly use # #

    def _gset_xmins_df_for_average(self):
        window, lag = self.xmin_qnty

        # TODO: add below printing to appropriate verbosity logging
        print("AVERAGE xmins to be calculated w/ rolling window size of "
              f"{window} days & lag days of {lag}")

        self._avbk_date = self.__get_back_date_label(window + lag)
        cx_df = self.__precompute_clauset_xmins()  # TODO: save2file for re-use
        self._dyn_cx_df = self.__dynamize_ts_df(cx_df, self._avbk_date)
        self.xmin_qnty = self._dyn_cx_df.rolling(window).mean().\
            shift(lag).loc[self.date_i:]

    def __precompute_clauset_xmins(self):
        overridden_opts = {'date_i': self._avbk_date,
                           'xmin_args': ('clauset', None),
                           'run_ks_test': False,
                           'compare_distros': False,
                           'plot_results': False}
        self._user_inputs.update(overridden_opts)
        pcs = Settings(self._user_inputs).settings
        pcs.data.stats_collabs = [('xmin', '')]
        pcs.data.stats_colname = "Clauset xmins"

        # TODO: add below printing to appropriate verbosity logging
        print("first need to pre-compute Clauset xmins b/w "
              f"[{pcs.data.date_i}, {pcs.data.date_f}]")

        from .analysis import DynamicAnalyzer
        analyzer = DynamicAnalyzer(pcs)
        analyzer.analyze()
        clauset_xmins_df = analyzer.get_resdf()
        clauset_xmins_df.columns = [f"{self.tst_map[t]} {grp}" for grp, t, _
                                    in clauset_xmins_df.columns]
        return clauset_xmins_df

    # ensures xmins for chosen ticker(s)/group(s) & tail(s) exist
    def _validate_xmins_df_statcols(self):
        assert self.xmin_rule in {'file', 'average'}

        from itertools import product
        needed_cols = [f"{st} {grp}" for st, grp in
                       product([self.tst_map[t] for t in self.tails_to_anal],
                               self.grouping_labs)]
        missing_cols = [nc for nc in needed_cols
                        if nc not in self.xmin_qnty.columns]
        if bool(missing_cols):
            raise ValueError(f"xmin columns {missing_cols} are needed but "
                             "not found in loaded xmins data")

    # # method exported to analysis.py, used strictly for logging # #
    def get_dyn_lbd(self, date):  # get dynamic lookback date
        d0 = date if self.approach == 'rolling' else self.date_i
        return self.__get_back_date_label(self._lookback, date0=d0)

    # # methods for creating the settings SimpleNamespace object(s) # #

    def _load_set_settings_config(self):
        SETTINGS_CFG = 'config/settings.yaml'  # TODO: refactor PATH into root
        with open(SETTINGS_CFG) as cfg:
            self._settings_config = yaml.load(cfg, Loader=yaml.SafeLoader)
        self._subsettings = list(self._settings_config.keys())
        if not self.plot_results:
            self._subsettings.remove('plot')

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
