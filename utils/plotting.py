# TODO: need to add boxplot, plotter using powerlaw's Fit API, barplot

from abc import ABC  # TODO: need to label @abstracmethod to work as ABC
from itertools import product
from string import Template

import json
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open('config/plotting/figures.yaml') as cfg:
    PLOT_CONFIG = yaml.load(cfg, Loader=yaml.SafeLoader)


class PlotDataCalculator:

    def __init__(self, settings, results):
        assert settings.anal.use_dynamic, \
            "static approach currently does not support plotting"
        self.sd = settings.data
        self.sa = settings.anal
        self.sp = settings.plot
        self.results = results
        self._gset_relevant_columns_resdf()
        #  self._gset_plot_data_df()
        self.update_ids = tuple(product(self.sd.grouping_labs,
                                        self.sa.tails_to_anal))

    def _gset_plot_data_df(self):
        pddf = self.__init_plot_data_df()
        pddf.update(self.rc_resdf)  # gets the (alpha, abs_len, ks_pv) series
        pddf.update(self.__calc_ci_bounds())  # calc the upper & lower bounds
        pddf.update(self.__calc_rel_len())  # calc the relative tail length
        #  print(pddf)
        #  print(pddf.columns)

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
        for grp, tail in self.update_ids:  # TODO: consider vectorizing
            alpha = self.rc_resdf[(grp, tail, 'alpha')]
            sigma = self.rc_resdf[(grp, tail, 'sigma')]
            delta = self.sp.alpha_quantile * sigma
            bounds[(grp, tail, 'upper')] = alpha + delta
            bounds[(grp, tail, 'lower')] = alpha - delta
        return bounds

    def __calc_rel_len(self):
        rel_lens = {}
        for grp, tail in self.update_ids:  # TODO: consider vectorizing
            rv_size = self.results[(grp, 'returns-statistics', '', 'count')]
            abs_len = self.rc_resdf[(grp, tail, 'abs_len')]
            rel_lens[(grp, tail, 'rel_len')] = abs_len / rv_size
        return rel_lens


# TODO: consider moving plotter state into own class
# and use this class only for plotting
class _BasePlotter(ABC):
    #  NOTE on method naming convention: excluding the special dunder methods,
    #  self-defined methods prepended by double underscores are meant to be
    #  called only by other private methods, which are themselves prepended by
    #  a single underscore.
    #  all __ methods have return values; while _ methods, not neccessarily
    """
    """

    #  def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):
    def __init__(self, settings, data, plot_type):  # fits_dict, data):
        """
        :param: ticker: string of ticker name
        :param: settings: SimpleNamespace object containing user-input options
        :param: data: dictionary of lists/arrays containing data to plot
        :param: plot_type: str; should be one of (αf, hg, ci, as, rs, ks, bx)
        """
        self.sd = settings.data
        self.sr = settings.rtrn
        self.sa = settings.anal
        self.sp = settings.plot
        self.data = data
        # TODO: make validator function for plot_type?
        self.ptyp = plot_type
        # NOTE: set the entire ptyp_config below if more config flags added
        #  self.ptyp_config = ptyp_config[self.ptyp]  # FIXME: module global
        #  self.multiplicities = PLOT_CONFIG[self.ptyp]["_multiplicities"]
        self.ax_title_base = (f"Time Period: {self.sd.date_i} "
                              f"- {self.sd.date_f}")
        #  NOTE: the fits_dict attr below now initialized in subclasses
        #  self.fits_dict = fits_dict["time_rolling"]

    # TODO: also add __repr__ method

    def __str__(self):
        # TODO: flesh this method out with better output
        return f"{self.__class__}, {self.__dict__}"

    # Methods for determining state-independent info; called in __init__()

    # NOTE: should be called before every _init_figure() call
    def _set_plotter_state(self, mult, tdir):
        """Sets the current state, i.e. the tail direction, plot
        type (CI, tail size, KS, etc.), and eventually ticker ID
        """
        self.curr_mult = mult  # multiplicity: must be 'single' OR 'double'
        # TODO: simplify naming to use either 'pos/neg' OR 'right/left'???
        self.curr_tdir = tdir
        self.curr_tsgn = "negative" if self.curr_tdir == "left" else "positive"
        self.curr_tsgs = self.curr_tsgn[:3]  # tail sign short form, ex. "pos"
        # TODO: below will be diff from self.ticker once unnested in tickers
        self.curr_ticker = self.ticker  # TODO: will be diff when plot unnested
        # TODO: consider adding if-check, to only update self.curr_ptinfo
        #       when stateful values inside of template_map changes
        self.curr_ptinfo = self.__set_ptyp_info()
        self.curr_vnames2plot = self.__get_vnames2plot()

    # State-aware and -dependent methods below

    # # state management and "bookkeeping" methods

    def __set_ptyp_info(self):
        """Named 'set' b/c there is dynamic info generated as well
        As opposed to simply fetching static data
        """

        sett = self.settings
        # TODO: make template_map subclass specific attribute?
        template_map = {
            "n_vec": sett.n_vec,
            "significance": sett.alpha_sgnf,
            "ticker": self.curr_ticker,
            "tdir": self.curr_tdir,
            "tsgn": self.curr_tsgn,
        }

        # NOTE: self.fits_dict is instantiated in subclass (problem?)
        ptyp_tmpl_dict = self.fits_dict[self.ptyp]
        ptyp_template = Template(json.dumps(ptyp_tmpl_dict))
        made_ptyp_info = ptyp_template.safe_substitute(template_map)

        return json.loads(made_ptyp_info)

    def __get_vnames2plot(self):
        """
        Set the correct data to be passed to _plot_vectors()
        """

        # TODO: refactor below to be more concise and DRY
        if self.curr_mult == "double":
            #  self.curr_tdir = "both"  # FIXME: where best to set this?
            tails_to_use = ("pos", "neg",)
        else:
            tails_to_use = (self.curr_tsgs,)

        return [f"{tsgs}_{vtyp}" for tsgs, vtyp in
                product(tails_to_use, self.curr_ptinfo["vec_types"])]

    # # methods for the actual plotting of the figure(s)
    # NOTE: plot phases partitioned into 3 for easy overwriting by subclasses

    def _init_figure(self):
        """Initialize a unique Matplotlib Figure instance,
        to set it up for the actual plotting, then returns it

        TODO: it should not care about the data being plotted nor the opts
        """
        # TODO: use fig, ax = plt.subplots() idiom to Initialize?
        fig = plt.figure(self.curr_ptinfo["fig_name"])
        axes_pos = (0.1, 0.20, 0.83, 0.70)
        ax = fig.add_axes(axes_pos)
        self.ax = ax

    @staticmethod
    def _set_line_style(vec_name):
        """Helper for setting the line style of the line plot
        :param: vec_name: string name of the vector to be plotted
        """
        if "pos" in vec_name:
            label = "Right tail"
            color = "green"
        elif "neg" in vec_name:
            label = "Left tail"
            color = "purple"
        # overwrite color and line when plotting α-bounds
        if "up_bound" in vec_name:
            label = "Upper bound"
            color = "black"
        elif "low_bound" in vec_name:
            label = "Lower bound"
            color = "blue"
        return {"label": label, "color": color}

    def _plot_vectors(self):
        """Given the data to plot, plot them onto the passed axes
        """

        # TODO: factor this into own function to keep DRY for histogram
        if extra_lines := self.curr_ptinfo.get("extra_lines", {}):
            vectors = extra_lines["vectors"]
            if isinstance(vectors, str):
                vectors = eval(vectors)
            for vec in vectors:
                # TODO: ensure all vecs are 2-tups for x-y plot??? (ex. hist)
                self.ax.plot(vec, **extra_lines["line_style"])

        for vn in self.curr_vnames2plot:
            # TODO: get line_style from self.curr_ptinfo first?
            self.ax.plot(self.data[vn], **self._set_line_style(vn))

    def _config_axes(self):
        """
        configure it appropriately after plotting (title, ticks, etc.)
        """

        sett = self.settings

        self.ax.set_title(self.curr_ptinfo["ax_title"] + self.ax_title_base)

        self.ax.set_xlim(xmin=0.0, xmax=sett.n_vec-1)
        self.ax.set_xticks(range(0, sett.n_spdt, sett.spec_labelstep))
        self.ax.set_xticklabels([dt[3:] for dt in
                                 sett.spec_dates[0::sett.spec_labelstep]],
                                rotation="vertical")

        self.ax.set_ylabel(self.curr_ptinfo["ax_ylabel"])

        self.ax.legend(**self.curr_ptinfo.get("ax_legend", {}))
        self.ax.grid()

    # NOTE: does this function need to be state aware?
    def _present_figure(self):  # , fig):  # , show_plot=False):
        """Show or save the plot(s) either save individual
        plots with fig.save, or show the plot(s) using plt.show()
        """
        # TODO: support interative modes

        if self.settings.show_plots:
            plt.show()

        if self.settings.save_plots:
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

        for mult, tdir in self.plot_combos:
            self._set_plotter_state(mult, tdir)
            self._init_figure()
            self._plot_vectors()
            self._config_axes()
            self._present_figure()


class TabledFigurePlotter(_BasePlotter):

    def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):

        # NOTE: maybe call super() after assigning self.fits_dict ??
        super().__init__(ticker, settings, data, plot_type)

        self.use_hist = True if self.ptyp == "hg" else False
        # FIXME: currently fits_dict below is a module global
        #  self.fits_dict = fits_dict["tabled_figure"]
        self.fits_dict = get_fits_dict()["tabled_figure"]
        # NOTE: problem :: fits_dict init'd in subclass, but curr_ptinfo is in
        #       parent; and it is only instantiated on __set_ptyp_info()
        #  self.table_info = self.curr_ptinfo["ax_table"]
        self.table_info = self.fits_dict[self.ptyp]["ax_table"]

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
            # TODO: ASK why use: h = 2*IQR/cuberoot(n_vec)
            h = 2 * IQR * np.power(self.settings.n_vec, -1/3)
            # TODO: xlim also uses max & min --> keep DRY
            n_bins = int((self.vec_max - self.vec_min)/h)
            hist_vals, bins, patches = self.ax.hist(self.data[vn],
                                                    n_bins, color="red")
            # FIXME: if multiple vecs in histogram, then hist_max below ovwrtn
            self.hist_max = np.max(hist_vals)

    def __plot_extra_hist_line(self):
        # TODO: factor this into own function to keep DRY for histogram

        extra_lines = self.curr_ptinfo["extra_lines"]
        vecs_encoded = extra_lines["vectors"]

        vecs_template = Template(json.dumps(vecs_encoded))
        vecs_str_tups = eval(vecs_template.substitute(vec_mean=self.vec_mean,
                                                      hist_max=self.hist_max))

        # NOTE: elements of vecs_str_tups are 2-tups to allow x vs. y plotting
        for x_str, y_str in vecs_str_tups:
            x, y = eval(x_str), eval(y_str)
            self.ax.plot(x, y, **extra_lines["line_style"])

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

    def _plot_vectors(self):
        """Given the data to plot, plot them onto the passed axes
        """

        if self.use_hist:
            self.__histogram()
            self.__plot_extra_hist_line()
        else:
            super()._plot_vectors()

    def _config_axes(self):

        if self.use_hist:
            self.ax.set_xlim(xmin=self.vec_min, xmax=self.vec_max)
            self.ax.set_ylabel(self.curr_ptinfo["ax_ylabel"])
            self.ax.set_ylim(ymin=0, ymax=self.hist_max)
            self.ax.legend()  # TODO: make legend & grid DRY
            self.ax.grid()
        else:
            super()._config_axes()

        self._add_table()


class TimeRollingPlotter(_BasePlotter):

    def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):

        super().__init__(ticker, settings, data, plot_type)
        # FIXME: currently fits_dict below is an imported module global
        # NOTE: self.fits_dict must 1st be instant'd as self.curr_ptinfo req it
        #  self.fits_dict = fits_dict["time_rolling"]
        self.fits_dict = get_fits_dict()["time_rolling"]
        # TODO: consider making fits_dict flat in plot_types level
        self.ax_title_base = (f"{self.ax_title_base}. "
                              f"Input: {self.return_type_label}")

    # NOTE: below is WIP --> maybe make into standalone module function
    def plot_ensemble(self):
        #  for ptyp in fits_dict["time_rolling"].keys():
        #      plotter = TimeRollingPlotter(ticker, settings, data, ptyp)
        #      plotter.plot()
        pass


# TODO: save a set of results data to do quick plot development!!!

tabled_figs = get_fits_dict()["tabled_figure"].keys()
def tabled_figure_plotter(ticker, settings, data):
    #  for ptyp in fits_dict["tabled_figure"].keys():
    for ptyp in tabled_figs:
        plotter = TabledFigurePlotter(ticker, settings, data, ptyp)
        plotter.plot()


timeroll_figs = get_fits_dict()["time_rolling"].keys()
def time_rolling_plotter(ticker, settings, data):
    #  for ptyp in fits_dict["time_rolling"].keys():
    for ptyp in timeroll_figs:
        plotter = TimeRollingPlotter(ticker, settings, data, ptyp)
        plotter.plot()
