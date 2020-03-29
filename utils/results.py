import numpy as np
import pandas as pd


class ResultsDataFrame:

    def __init__(self, settings):
        self.ctrl = settings.ctrl
        self.data = settings.data
        self.anal = settings.anal

    def _init_static(self):
        self._gixn = self.data.grouping_type  # gixn: grouping index name
        n_groupings = len(self.data.grouping_labs)
        ridx_name = (self._gixn if n_groupings == 1 else self._gixn.pluralize()
                     if self.anal.use_static else self.data.anal_dates.name)
        ridx_labs = (self.data.grouping_labs if self.anal.use_static else
                     self.data.anal_dates)
        ridx = pd.Index(ridx_labs, name=ridx_name)
        # ridx: ROW index above, & cidx: COLUMN index below
        cidx = pd.MultiIndex.from_tuples(self.data.stats_collabs,
                                         names=(self.data.stats_colname, ''))

        df_tail = pd.DataFrame(np.full((len(ridx), len(cidx)), np.nan),
                               index=ridx, columns=cidx, dtype=float)

        # TODO: use the special str subclass w/ .pluralize for name below?
        tidx_name = 'tails' if len(self.anal.tails_to_anal) == 2 else 'tail'
        return pd.concat({t: df_tail for t in self.anal.tails_to_anal},
                         axis=1, names=(tidx_name,))

    # TODO look into pd.concat alternatives
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

    def _init_dynamic(self):
        df_sub = self._init_static()
        return pd.concat({sub: df_sub for sub in self.data.grouping_labs},
                         axis=1, names=(self._gixn,))

    def initialize(self):
        return (self._init_static() if self.anal.use_static else
                self._init_dynamic())
