import numpy as np
import pandas as pd


class ResultsDataFrame:

    def __init__(self, settings):
        self.ctrl = settings.ctrl
        self.data = settings.data
        self.anal = settings.anal

    def __get_row_idxname(self):
        idxname_map = {None: 'Tickers',
                       'country': 'Countries',
                       'maturity': 'Maturities',
                       'region': 'Regions'}
        return idxname_map[self.anal.partition]

    def _init_results_df_dynamic(self):
        #  df_sub = super(DynamicAnalyzer, self)._init_results_df()
        #  return pd.concat({sub: df_sub for sub in self.data.grouping_labs},
        #                   axis=1)  # TODO: set column index level names
        pass

    def initialize(self):
        ridx_labs = (self.data.grouping_labs if self.anal.use_static
                     else self.data.anal_dates)
        ridx_name = (self.__get_row_idxname() if self.anal.use_static
                     else self.data.anal_dates.name)
        ridx = pd.Index(ridx_labs, name=ridx_name)
        # ridx: ROW index above, & cidx: COLUMN index below
        cidx = pd.MultiIndex.from_tuples(self.data.stats_collabs,
                                         names=(self.data.stats_colname, ''))

        df_tail = pd.DataFrame(np.full((len(ridx), len(cidx)), np.nan),
                               index=ridx, columns=cidx, dtype=float)

        tail_cidx_name = 'Tails' if self.anal.n_tails == 2 else 'Tail'
        return pd.concat({t: df_tail for t in self.anal.tails_to_use},
                         axis=1, names=(tail_cidx_name,))
        # TODO look into pd.concat alternatives
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
