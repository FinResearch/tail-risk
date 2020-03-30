import numpy as np
import pandas as pd


class Results:

    def __init__(self, settings):
        #  self.sc = settings.ctrl  # doesn't appear to need
        self.sd = settings.data
        self.sa = settings.anal

        self.df = self.initialize()

    def _init_static(self):
        # gixn: grouping index name
        self._gixn = self.sd.grouping_type
        # ridx: ROW index
        ridx_name = (self.sd.anal_dates.name if self.sa.use_dynamic else
                     self._gixn if len(self.sd.grouping_labs) == 1 else
                     self._gixn.pluralize())
        ridx_labs = (self.sd.anal_dates if self.sa.use_dynamic
                     else self.sd.grouping_labs)
        ridx = pd.Index(ridx_labs, name=ridx_name)
        # cidx: COLUMN index below
        cidx = pd.MultiIndex.from_tuples(self.sd.stats_collabs,
                                         names=(self.sd.stats_colname, ''))

        df_tail = pd.DataFrame(np.full((len(ridx), len(cidx)), np.nan),
                               index=ridx, columns=cidx, dtype=float)

        # TODO: use the special str-subclass w/ .pluralize for tail name below?
        tidx_name = 'tails' if len(self.sa.tails_to_anal) == 2 else 'tail'
        return pd.concat({t: df_tail for t in self.sa.tails_to_anal},
                         axis=1, names=(tidx_name,))

    # TODO look into pd.concat alternatives
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

    def _init_dynamic(self):
        df_sub = self._init_static()
        return pd.concat({sub: df_sub for sub in self.sd.grouping_labs},
                         axis=1, names=(self._gixn,))

    def initialize(self):
        return (self._init_dynamic() if self.sa.use_dynamic else
                self._init_static())
