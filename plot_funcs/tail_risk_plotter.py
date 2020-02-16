from abc import ABC  # TODO: label @abstracmethod
from itertools import product
from string import Template

import json

import numpy as np
import matplotlib.pyplot as plt

# NOTE: currently module globals
#  from plot_funcs.fits_dict import fits_dict, ptyp_config
from plot_funcs.fits_dict import get_fits_dict, get_ptyp_config
# TODO: consider making class for these, and pass them as dict objects


# TODO: consider making values returned from this function part
# of plot_types_static_info (ptsi) data --> now: self.curr_ptinfo
# TODO: alternatively, make this into @staticmethod of TimeRollingPlotter
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


# TODO: consider moving plotter state into own class
# and use this class only for plotting
class TailRiskPlotter(ABC):
    """
    Note on method name conventions: other than the reserved dunder methods,
    self-defined methods prepended by a double underscore are meant to be
    called only by other private methods, which are themselves prepended by
    a single underscore.

    all __METHOD's also have a return value
    while _METHOD's do not neccessarily
    """

    def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):
        """
        :param: ticker: string of ticker name
        :param: settings: SimpleNamespace object containing user-input options
        :param: data: dictionary of lists/arrays containing data to plot
        :param: plot_type: str; should be one of (αf, hg, ci, as, rs, ks, bx)
        """
        self.ticker = ticker
        self.settings = settings
        # TODO: consider passing in only the data needed by the given plot_type
        self.data = data
        # TODO: make validator function for plot_type?
        self.ptyp = plot_type
        # NOTE: set the entire ptyp_config below if more config flags added
        #  self.ptyp_config = ptyp_config[self.ptyp]  # FIXME: module global
        #  self.multiplicities = ptyp_config[self.ptyp]["multiplicities"]
        self.multiplicities = get_ptyp_config()[self.ptyp]["multiplicities"]
        self.tails_used = self.__get_tails_used()
        self.plot_combos = self.__get_plot_combos()
        self.return_type_label = self.__get_return_type_label()
        self.ax_title_base = (f"Time Period: {self.settings.date_i} "
                              f"- {self.settings.date_f}")
        #  NOTE: the fits_dict attr below now initialized in subclasses
        #  self.fits_dict = fits_dict["time_rolling"]

    # TODO: also add __repr__ method

    def __str__(self):
        # TODO: flesh this method out with better output
        return f"{self.__class__}, {self.__dict__}"

    # Methods for determining state-independent info; called in __init__()

    def __get_tails_used(self):
        """Return tuple containing the tails selected/used
        """

        tails_used = []
        if self.settings.use_right_tail:
            tails_used.append("right")
        if self.settings.use_left_tail:
            tails_used.append("left")

        return tuple(tails_used)

    # TODO: consider moving this method into child class
    def __get_plot_combos(self):
        """Return tuple of 2-tups representing all concrete figures requested
        """

        if len(self.tails_used) == 2:
            mults = self.multiplicities
            # TODO: set fig_name to state "both tails" instead of either R/L
            #  tails = ("right",)  # NOTE: can also use "left", does not matter
            # FIXME: above misses ("single", "left") when using both tails
            tails = ("left",)  # TODO: this was to test "left" works
        else:
            # NOTE: if single tail used, then mult is necessarily "singles"
            mults = ("singles",)
            tails = self.tails_used

        return product(mults, tails)

    def __get_return_type_label(self):
        """This info is independent of the state (ticker, tail, etc.).
        Instead it is solely determined by the chosen return type
        """

        pt_i = "P(t)"
        pt_f = f"P(t+{self.settings.tau})"

        if self.settings.return_type == "basic":
            label = f"{pt_f} - {pt_i}"
        elif self.settings.return_type == "relative":
            label = f"{pt_f}/{pt_i} - 1.0"
        elif self.settings.return_type == "log":
            label = rf"$\log$({pt_f}/{pt_i})"

        if self.settings.absolutize:
            label = f"|{label}|"

        return label

    # NOTE: should be called before every _init_figure() call
    def _set_plotter_state(self, mult, tdir):
        """Sets the current state, i.e. the tail direction, plot
        type (CI, tail size, KS, etc.), and eventually ticker ID
        """
        self.curr_mult = mult  # multiplicity: must be 'single' OR 'double'
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
            "significance": sett.significance,
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
            self.ax.plot(self.data[vn], **_set_line_style(vn))

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


class TabledFigurePlotter(TailRiskPlotter):

    def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):

        # NOTE: maybe call super() after assigning self.fits_dict ??
        super(TabledFigurePlotter, self).__init__(ticker, settings,
                                                  data, plot_type)

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
            super(TabledFigurePlotter, self)._plot_vectors()

    def _config_axes(self):

        if self.use_hist:
            self.ax.set_xlim(xmin=self.vec_min, xmax=self.vec_max)
            self.ax.set_ylabel(self.curr_ptinfo["ax_ylabel"])
            self.ax.set_ylim(ymin=0, ymax=self.hist_max)
            self.ax.legend()  # TODO: make legend & grid DRY
            self.ax.grid()
        else:
            super(TabledFigurePlotter, self)._config_axes()

        self._add_table()


class TimeRollingPlotter(TailRiskPlotter):

    def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):

        super(TimeRollingPlotter, self).__init__(ticker, settings,
                                                 data, plot_type)
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
