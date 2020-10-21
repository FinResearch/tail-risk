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
        #  self.sr = settings.rtrn
        #  self.sa = settings.anal
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

    def __sub_figure_metadata(self, fig_metdat_temp):
        """
        """
        self._gset_tail_metadata()
        temp_sub_map = {"vec_size": self.sp.vec_size,
                        "conf_lvl": self.sp.confidence_level,
                        "timeperiod": self.sp.title_timeperiod,
                        "rtrn_label": self.sp.returns_label,
                        "label": f'{self.sd.grouping_type}: {self.plt_lab}',
                        "tdir": self.tdir,
                        "tsgn": self.tsgn}
        fmdat_template = Template(json.dumps(fig_metdat_temp))
        fmdat_complete = fmdat_template.safe_substitute(temp_sub_map)
        return json.loads(fmdat_complete)

    # # methods for the actual plotting of the figure(s)
    # NOTE: plot phases partitioned into 3 for easy overwriting by subclasses

    def _init_figure(self):
        """Initialize a unique Matplotlib Figure instance,
        to set it up for the actual plotting, then returns it

        TODO: it should not care about the data being plotted nor the opts
        """
        # TODO: use fig, ax = plt.subplots() idiom to Initialize?
        fig = plt.figure(self.fmdat["fig_name"])
        axes_pos = (0.1, 0.20, 0.83, 0.70)
        ax = fig.add_axes(axes_pos)
        self.ax = ax

    @staticmethod
    def _set_line_style(vec_id):
        """Helper for setting the line style of the line plot
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

    def _plot_vectors(self):
        """Given the data to plot, plot them onto the passed axes
        """

        # TODO: factor this into own function to keep DRY for histogram
        # TODO: ensure all vecs are 2-tups for x-y plot?? (ex. hist)
        if extra_lines := self.fmdat.get("extra_lines", {}):
            extra_vecs = [eval(vec_str) for vec_str in extra_lines['vectors']]
            for vec in extra_vecs:
                self.ax.plot(vec, **extra_lines['line_style'])

        for tail in self.tail_tup:
            for st in self.fmdat['stats2plt']:
                vec_id = (self.plt_lab, tail, st)
                self.ax.plot(self.pcalc.get_vec(vec_id),
                             **self._set_line_style(vec_id))

    def _config_axes(self):
        """
        configure it appropriately after plotting (title, ticks, etc.)
        """
        self.ax.set_title(self.fmdat["ax_title"])
        self.ax.set_xlim(xmin=0.0, xmax=self.sp.vec_size-1)

        self.ax.set_xlim(xmin=0.0, xmax=sett.n_vec-1)
        self.ax.set_xticks(range(0, sett.n_spdt, sett.spec_labelstep))
        self.ax.set_xticklabels([dt[3:] for dt in
                                 sett.spec_dates[0::sett.spec_labelstep]],
                                rotation="vertical")

        self.ax.set_ylabel(self.fmdat["ax_ylabel"])

        self.ax.legend(**self.fmdat.get("ax_legend", {}))
        self.ax.grid()

    # NOTE: does this function need to be state aware?
    def _present_figure(self):  # , fig):  # , show_plot=False):
        """Show or save the plot(s) either save individual
        plots with fig.save, or show the plot(s) using plt.show()
        """
        # TODO: support interative modes

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
        self._plot_vectors()
        self._config_axes()
        self._present_figure()


class TabledFigurePlotter(_BasePlotter):

    #  def __init__(self, settings, plot_label, plot_combo_id,
    #               figure_metadata, plot_data_calc):
    #      super().__init__(settings, plot_label, plot_combo_id,
    #                       figure_metadata, plot_data_calc)

    def __gen_table_text(self):

        # text generating functions; use dict.pop to remove non-table-kwarg
        tgfuncs = eval(self.table_info.pop("_cellText_gens"))

        extra_cell = self.table_info.pop("_extra_cell", ())

        cellText = []
        for i, vn in enumerate(self.curr_vnames2plot):
            row_cells = [np.round(fn(self.data[vn]), 4) for fn in tgfuncs]
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

    #  def _plot_vectors(self):
    #      """Given the data to plot, plot them onto the passed axes
    #      """
    #
    #      if self.use_hist:
    #          self.__histogram()
    #          self.__plot_extra_hist_line()
    #      else:
    #          super()._plot_vectors()

    def _config_axes(self):
        super()._config_axes()
        self._add_table()


class HistogramPlotter(TabledFigurePlotter):

    #  def __init__(self, settings, plot_label, plot_combo_id,
    #               figure_metadata, plot_data_calc):
    #      super().__init__(settings, plot_label, plot_combo_id,
    #                       figure_metadata, plot_data_calc)
    #      assert self.plt_typ.value == 'hg'

    def __calc_vec_stats(self, vec):
        self.vec_min = np.min(vec)
        self.vec_max = np.max(vec)
        self.vec_mean = np.mean(vec)

    def __histogram(self):

        npp = np.percentile  # NOTE: shorthand for NumPy method

        # TODO: is for-loop necessary if each hist only contains a single vec?
        for vn in self.curr_vnames2plot:
            # FIXME: if multiple vecs in histogram, then self attrs
            #        calc'd below gets overwritten
            self.__calc_vec_stats(self.data[vn])

            IQR = npp(self.data[vn], 75) - npp(self.data[vn], 25)
            # TODO: ASK why use: h = 2*IQR/cuberoot(vec_size)
            h = 2 * IQR * np.power(self.sp.vec_size, -1/3)
            # TODO: xlim also uses max & min --> keep DRY
            n_bins = int((self.vec_max - self.vec_min)/h)
            hist_vals, bins, patches = self.ax.hist(self.data[vn],
                                                    n_bins, color="red")
            # FIXME: if multiple vecs in histogram, then hist_max below ovwrtn
            self.hist_max = np.max(hist_vals)

    def __plot_extra_hist_line(self):
        # TODO: factor this into own function to keep DRY for histogram

        extra_lines = self.fmdat["extra_lines"]
        vecs_encoded = extra_lines["vectors"]

        vecs_template = Template(json.dumps(vecs_encoded))
        vecs_str_tups = eval(vecs_template.substitute(vec_mean=self.vec_mean,
                                                      hist_max=self.hist_max))

        # NOTE: elements of vecs_str_tups are 2-tups to allow x vs. y plotting
        for x_str, y_str in vecs_str_tups:
            x, y = eval(x_str), eval(y_str)
            self.ax.plot(x, y, **extra_lines["line_style"])

    def _plot_vectors(self):
        """Given the data to plot, plot them onto the passed axes
        """
        self.__histogram()
        self.__plot_extra_hist_line()

    def _config_axes(self):
        self.ax.set_xlim(xmin=self.vec_min, xmax=self.vec_max)
        self.ax.set_ylabel(self.fmdat["ax_ylabel"])
        self.ax.set_ylim(ymin=0, ymax=self.hist_max)
        self.ax.legend()  # TODO: make legend & grid DRY
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

        #  Plot_cls = eval(fig_meta['Plot_cls'])
        Plot_cls = TimeRollingPlotter

        for label in settings.data.grouping_labs:
            plotter = Plot_cls(settings, label, combo_id,
                               fig_meta, plot_data_calc)
            plotter.plot()
            if ptid == 'bx':
                break
