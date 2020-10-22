# TODO: need to add boxplot, plotter using powerlaw's Fit API, barplot

from abc import ABC  # TODO: need to label @abstracmethod to work as ABC
from itertools import product
from string import Template

import json
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PlotDataCalculator:

    def __init__(self, settings, results):
        assert settings.anal.use_dynamic, \
            "static approach currently does not support plotting"
        self.sd = settings.data
        self.sa = settings.anal
        self.sp = settings.plot
        self.results = results
        self._update_ids = tuple(product(self.sd.grouping_labs,
                                         self.sa.tails_to_anal))
        self._gset_relevant_columns_resdf()
        self._gset_plot_data_df()

    def _gset_relevant_columns_resdf(self):
        # NOTE: tstats below must match those listed under 'tail-statistics'
        #       subsection in config/output_columns.yaml
        reqd_tstats = {'alpha', 'sigma', 'abs_len'}
        if self.sa.run_ks_test:
            reqd_tstats.add('ks_pv')
        rc = [idx for idx in self.results.columns if idx[-1] in reqd_tstats]
        self.rc_resdf = self.results[rc]  # rc: relevant columns
        if 'tail-statistics' in self.rc_resdf.columns.levels[-2]:
            cidx = self.rc_resdf.columns
            self.rc_resdf.columns = cidx.droplevel(-2)
        else:
            raise ValueError('there is a column MultiIndex error!')

    def __init_plot_data_df(self):
        ridx = self.sd.anal_dates
        plot_data_stats = ['alpha', 'upper', 'lower', 'abs_len', 'rel_len']
        if self.sa.run_ks_test:
            plot_data_stats.append('ks_pv')
        cidx = pd.MultiIndex.from_product([self.sd.grouping_labs,
                                           self.sa.tails_to_anal,
                                           plot_data_stats])
        return pd.DataFrame(np.full((len(ridx), len(cidx)), np.nan),
                            index=ridx, columns=cidx, dtype=float)

    def __calc_ci_bounds(self):
        bounds = {}
        for grp, tail in self._update_ids:  # TODO: consider vectorizing
            alpha = self.rc_resdf[(grp, tail, 'alpha')]
            sigma = self.rc_resdf[(grp, tail, 'sigma')]
            delta = self.sp.alpha_quantile * sigma
            bounds[(grp, tail, 'upper')] = alpha + delta
            bounds[(grp, tail, 'lower')] = alpha - delta
        return bounds

    def __calc_rel_len(self):
        rel_lens = {}
        for grp, tail in self._update_ids:  # TODO: consider vectorizing
            rv_size = self.results[(grp, 'returns-statistics', '', 'count')]
            abs_len = self.rc_resdf[(grp, tail, 'abs_len')]
            rel_lens[(grp, tail, 'rel_len')] = abs_len / rv_size
        return rel_lens

    def _gset_plot_data_df(self):
        self.pddf = self.__init_plot_data_df()
        self.pddf.update(self.rc_resdf)  # gets (alpha, abs_len, ks_pv) vecs
        self.pddf.update(self.__calc_ci_bounds())  # get upper & lower bounds
        self.pddf.update(self.__calc_rel_len())  # get relative tail length

    def get_vec(self, vec_id, rtn_ndarray=True):
        # NOTE: vec_id must be of the form: (group_label, Tail, statistic)
        vec_series = self.pddf[vec_id]
        return vec_series.to_numpy() if rtn_ndarray else vec_series


class _BasePlotter(ABC):
    """
    """

    def __init__(self, settings, plot_label, plot_combo_id,
                 figure_metadata, plot_data_calc):
        """
        :param: settings: SimpleNamespace object containing user-input options
        :param: plot_label: string; ticker or group ID
        :param: plot_combo_id: tuple; ex. (PlotType.ALPHA_FIT, (Tail.right,))
        :param: figure_metadata: figure metadata dict for a given plot type
        :param: plot_data_calc: PlotDataCalculator object instance
        """
        self.sd = settings.data
        self.sp = settings.plot
        self.plt_lab = plot_label
        self.plt_typ, self.tail_tup = plot_combo_id
        # TODO: import PlotType enum class to remove redundant assignment below
        self.plt_id = self.plt_typ.value
        self.fmdat = self.__sub_figure_metadata(figure_metadata)
        self.pcalc = plot_data_calc

    # TODO: also add __repr__ method
    def __str__(self):
        # TODO: flesh this method out with better output
        return f"{self.__class__}, {self.__dict__}"

    def _gset_tail_metadata(self):
        _tail_sgl = self.tail_tup[0] if len(self.tail_tup) == 1 else None
        self.tdir = f'{_tail_sgl.name} tail' if _tail_sgl else 'both tails'
        _tsgn_map = {1: 'positive', -1: 'negative'}
        self.tsgn = _tsgn_map[_tail_sgl.value] if _tail_sgl else ''

    @staticmethod
    def _substitute_json_template_(json_temp, sub_map):
        template = Template(json.dumps(json_temp))
        return json.loads(template.safe_substitute(sub_map))

    def __sub_figure_metadata(self, fig_metdat_temp):
        """
        """
        self._gset_tail_metadata()
        fig_sub_map = {"vec_size": self.sp.vec_size,
                       "conf_lvl": self.sp.confidence_level,
                       "timeperiod": self.sp.title_timeperiod,
                       "rtrn_label": self.sp.returns_label,
                       "label": f'{self.sd.grouping_type}: {self.plt_lab}',
                       "tdir": self.tdir,
                       "tsgn": self.tsgn}
        return self._substitute_json_template_(fig_metdat_temp, fig_sub_map)

    # # methods for the actual plotting of the figures

    @staticmethod
    def __set_line_style(vec_id):
        """Helper func for setting the line style of the line plot
        :param: vec_id: unique 3-tuple that IDs any given vector
        """
        _, tail, stat = vec_id
        use_right_tail = tail.value == 1
        label = 'Right tail' if use_right_tail else 'Left tail'
        color = 'green' if use_right_tail else 'purple'
        # overwrite color and line when plotting α-bounds
        if stat == 'upper':
            label = "Upper bound"
            color = "black"
        elif stat == 'lower':
            label = "Lower bound"
            color = "blue"
        return {'label': label, 'color': color}

    def _init_figure(self):
        """Initialize a unique Matplotlib Figure instance used
        for the actual plotting, then sets it onto self
        """
        # TODO: use fig, ax = plt.subplots() idiom to Initialize?
        fig = plt.figure(self.fmdat["fig_name"])
        axes_pos = (0.1, 0.20, 0.83, 0.70)
        ax = fig.add_axes(axes_pos)
        self.ax = ax

    def _gset_vec2plt(self):
        self.v2p_map = {}
        for tail in self.tail_tup:
            for st in self.fmdat['stats2plt']:
                vec_id = (self.plt_lab, tail, st)
                self.v2p_map[vec_id] = (self.pcalc.get_vec(vec_id))

    def _plot_vectors(self):
        """Given the data to plot, plot them onto the passed axes
        """
        # TODO: factor this into own function to keep DRY for histogram
        # TODO: ensure all vecs are 2-tups for x-y plot?? (ex. hist)
        if extra_lines := self.fmdat.get("extra_lines", {}):
            extra_vecs = [eval(vec_str) for vec_str in extra_lines['vectors']]
            for vec in extra_vecs:
                self.ax.plot(vec, **extra_lines['line_style'])
        for vid, vec in self.v2p_map.items():
            self.ax.plot(vec, **self.__set_line_style(vid))

    def _config_axes(self):
        """Configure it appropriately after plotting (title, ticks, etc.)
        """
        self.ax.set_title(self.fmdat["ax_title"])
        self.ax.set_xlim(xmin=0.0, xmax=self.sp.vec_size-1)
        self.ax.set_xticks(self.sp.xtick_locs)
        self.ax.set_xticklabels(self.sp.xtick_labs, rotation="vertical")
        # TODO: take only DD-MM for xtick labels OR something else?
        self.ax.set_ylabel(self.fmdat["ax_ylabel"])
        self.ax.legend(**self.fmdat.get("ax_legend", {}))
        self.ax.grid()

    def _present_figure(self):
        """Shows and/or saves the generated plot
        """

        if self.sp.show_plots:
            plt.show()

        if self.sp.save_plots:
            pass
        else:
            # TODO: implement plot saving functionality here
            pass

    # TODO: add *methods parameters to be optionally called?
    def plot(self):
        """
        This is the publicly exposed API to this class.
        Just initialize a plotter object, and call plotter.plot()
        """
        self._init_figure()
        self._gset_vec2plt()
        self._plot_vectors()
        self._config_axes()
        self._present_figure()


class TabledFigurePlotter(_BasePlotter):

    def __gen_table_text(self):
        self.table_info = self.fmdat['ax_table']
        # text generating functions; use dict.pop to remove non-table-kwarg
        tgen_funcs = [eval(fn_str) for fn_str in
                      self.table_info.pop("_cellText_gens")]
        extra_cell = self.table_info.pop("_extra_cell", ())
        cellText = []
        for i, vec in enumerate(self.v2p_map.values()):
            row_cells = [np.round(fn(vec), 4) for fn in tgen_funcs]
            if extra_cell:
                cell_val, pos = extra_cell[i]
                row_cells.insert(pos, cell_val)
            cellText.append(row_cells)
        return cellText

    def _add_table(self):

        cellText = self.__gen_table_text()
        # TODO: attach Table object to self?
        table = self.ax.table(cellText=cellText, **self.table_info)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(0.5, 0.5)

    def _config_axes(self):
        super()._config_axes()
        self._add_table()


class HistogramPlotter(TabledFigurePlotter):

    def _gset_vec_stats(self, vec):
        self.vec_min = np.min(vec)
        self.vec_max = np.max(vec)
        self.vec_mean = np.mean(vec)

    def _histogram(self):
        npp = np.percentile  # NOTE: shorthand for NumPy method
        for vec in self.v2p_map.values():  # TODO: need for loop if only 1 vec?
            # FIXME: if multiple vecs in histogram, then self attrs
            #        calc'd below gets overwritten
            self._gset_vec_stats(vec)
            IQR = npp(vec, 75) - npp(vec, 25)
            # TODO: ASK why use: h = 2*IQR/cuberoot(vec_size)
            h = 2 * IQR * np.power(self.sp.vec_size, -1/3)
            n_bins = int((self.vec_max - self.vec_min)/h)
            hist_vals, bins, patches = self.ax.hist(vec, n_bins, color="red")
            # FIXME: if multiple vecs in histogram, then hist_max below ovwrtn
            self.hist_max = np.max(hist_vals)
        self.hist_sub_map = {'vec_mean': self.vec_mean,
                             'hist_max': self.hist_max}

    def _plot_extra_hist_line(self):
        # TODO: factor plt_extra_line in parent cls into own func to keep DRY?
        extra_line = self.fmdat['extra_line']
        xy_vec_strs = self._substitute_json_template_(extra_line['vector'],
                                                      self.hist_sub_map)
        x, y = [eval(c) for c in xy_vec_strs]
        self.ax.plot(x, y, **extra_line['line_style'])

    def _plot_vectors(self):
        """Overwrite _BasePlotter's _plot_vectors method to plot the histogram
        """
        self._histogram()
        self._plot_extra_hist_line()

    def _config_axes(self):
        self.ax.set_xlim(xmin=self.vec_min, xmax=self.vec_max)
        self.ax.set_ylabel(self.fmdat['ax_ylabel'])
        self.ax.set_ylim(ymin=0, ymax=self.hist_max)
        self.ax.legend()
        self.ax.grid()
        self._add_table()


class TimeRollingPlotter(_BasePlotter):

    def __init__(self, settings, plot_label, plot_combo_id,
                 figure_metadata, plot_data_calc):
        super().__init__(settings, plot_label, plot_combo_id,
                         figure_metadata, plot_data_calc)


def plot_ensemble(settings, results):
    with open('config/plotting/figures.yaml') as cfg:
        figs_meta_map = yaml.load(cfg, Loader=yaml.SafeLoader)
    plot_data_calc = PlotDataCalculator(settings, results)
    for combo_id in settings.plot.plot_combos:
        ptid = combo_id[0].value
        fig_meta = figs_meta_map[ptid]
        Plot_cls = eval(fig_meta['Plot_cls'])
        for label in settings.data.grouping_labs:
            plotter = Plot_cls(settings, label, combo_id,
                               fig_meta, plot_data_calc)
            plotter.plot()
            if ptid == 'bx':
                break
