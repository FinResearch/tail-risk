from abc import ABC
from itertools import product
from string import Template

import json

import numpy as np
import matplotlib.pyplot as plt

# NOTE: currently module globals
from plot_funcs.fits_dict import fits_dict
# TODO: consider making class for these, and pass them as dict objects


# TODO: consider making values returned from this function part
# of plot_types_static_info (ptsi) data --> now: self.ptyp_info
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
    """

    def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):
        """
        :param: ticker: string of ticker name
        :param: settings: SimpleNamespace object containing user-input options
        :param: fits_dict: figure information templates dict
        :param: data: dictionary of lists/arrays containing data to plot
        :param: plot_type: str; should be one of (αf, hg, ci, as, rs, ks, bx)
        """
        self.ticker = ticker
        self.settings = settings
        # TODO: consider passing in only the data needed by the plot_type
        self.data = data
        # TODO: make validator function for plot_type?
        self.ptyp = plot_type
        self.tails_used = self.__get_tails_used()
        self.return_type_label = self.__get_return_type_label()
        self.ax_title_base = (f"Time Period: {self.settings.date_i} "
                              f"- {self.settings.date_f}. "
                              f"Input: {self.return_type_label}")
        #  NOTE: the fits_dict attr below now initialized in subclasses
        #  self.fits_dict = fits_dict["time_rolling"]
        #  self.all_plot_combos = self.__get_all_plot_combos()
        #  NOTE: the 2 attr above are initialized in the child class

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
    def _get_all_plot_combos(self):
        """Return tuple of 2-tups representing all concrete figures requested
        """
        # TODO: i.e. when plt_typ is one of ["as", "rs", "ks"],
        # then need to also do a combined fig of pos+neg tails
        return tuple(product(self.tails_used, self.fits_dict.keys()))
        # NOTE: consider only generating plot combos for curr_ptinfo?
        #       b/c self.fits_dict is no longer init'd in this parent class

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
    def _set_plotter_state(self, tdir, ptyp):
        """Sets the current state, i.e. the tail direction, plot
        type (CI, tail size, KS, etc.), and eventually ticker ID
        """
        self.curr_tdir = tdir
        self.curr_ptyp = ptyp
        self.curr_tsgn = "negative" if self.curr_tdir == "left" else "positive"
        self.curr_tsgs = self.curr_tsgn[:3]  # tail sign short form, ex. "pos"
        # TODO: below will be diff from self.ticker once unnested in tickers
        self.curr_ticker = self.ticker  # TODO: will be diff when plot unnested
        # TODO: consider adding if-check, to only update self.ptyp_info
        #       when stateful values inside of template_map changes
        self.ptyp_info = self.__set_ptyp_info()

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

        ptyp_tmpl_dict = self.fits_dict[self.ptyp]
        ptyp_template = Template(json.dumps(ptyp_tmpl_dict))
        made_ptyp_info = ptyp_template.safe_substitute(template_map)

        return json.loads(made_ptyp_info)

    def __get_vecs2plot(self):
        """
        Set the correct data to be passed to _plot_lines()
        """
        # TODO: use this function to do processing on vecs2plot?
        # For example, when plt_typ is one of ["as", "rs", "ks"],
        # then do a combined plot of pos + neg

        # TODO: consider making curr_tsgs into a tuple instead of just a str
        # so that ("right",) or ("right", "left") can be producted w/ vec_types
        return [f"{self.curr_tsgs}_{ptyp}" for ptyp
                in self.curr_ptinfo["vec_types"]]

    # # methods for the actual plotting of the figure(s)
    # NOTE: how much state do these methods now depend on, now that
    #       ptyp_info has been moved into __init__()?

    def _init_figure(self):
        """Initialize a unique Matplotlib Figure instance,
        to set it up for the actual plotting, then returns it

        TODO: it should not care about the data being plotted nor the opts
        """

        # TODO: use fig, ax = plt.subplots() idiom to Initialize?
        fig = plt.figure(self.ptyp_info["fig_name"])
        axes_pos = (0.1, 0.20, 0.83, 0.70)
        ax = fig.add_axes(axes_pos)

        # TODO: consider attaching to self as self.curr_axes ???
        return ax

    # TODO: fold this method into _init_figure()?
    # since all figure specific info are stored inside of fits_dict
    def _plot_lines(self, ax):  # , vec_names):
        """Given the data to plot, plot them onto the passed axes
        """

        vecs2plot = self.__get_vecs2plot()

        if extra_lines := self.ptyp_info.get("extra_lines", {}):
            vectors = extra_lines["vectors"]
            if isinstance(vectors, str):
                vectors = eval(vectors)
            for vec in vectors:
                # TODO: ensure all vecs are 2-tuples to allow x vs. y plotting
                ax.plot(vec, **extra_lines["line_style"])

        for vn in vecs2plot:
            # TODO: try get the line_style from self.ptyp_info first
            ax.plot(self.data[vn], **_set_line_style(vn))

    # TODO: fold this method into _init_figure()?
    # since all figure specific info are stored inside of fits_dict
    def _config_axes(self, ax):
        """
        configure it appropriately after plotting (title, ticks, etc.)
        """

        sett = self.settings

        ax.set_title(self.ptyp_info["ax_title"] + self.ax_title_base)

        ax.set_xlim(xmin=0.0, xmax=sett.n_vec-1)
        ax.set_xticks(range(0, sett.n_spdt, sett.spec_labelstep))
        ax.set_xticklabels([dt[3:] for dt in
                            sett.spec_dates[0::sett.spec_labelstep]],
                           rotation="vertical")

        ax.set_ylabel(self.ptyp_info["ax_ylabel"])

        ax.legend(**self.ptyp_info.get("ax_legend", {}))
        ax.grid()

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

        for tdir, ptyp in self.all_plot_combos:
            self._set_plotter_state(tdir, ptyp)
            ax = self._init_figure()
            self._plot_lines(ax)
            self._config_axes(ax)
            self._present_figure()


class TabledFigurePlotter(TailRiskPlotter):

    def __init__(self, ticker, settings, data):  # fits_dict, data):

        super(TimeRollingPlotter, self).__init__(ticker, settings, data)

        # FIXME: currently fits_dict below is a module global
        self.fits_dict = fits_dict["tabled_figure"]
        self.all_plot_combos = self._get_all_plot_combos()

    # NOTE: below is WIP
    def add_table(self):
        pass

    def plot(self):
        pass


class TimeRollingPlotter(TailRiskPlotter):

    def __init__(self, ticker, settings, data, plot_type):  # fits_dict, data):

        super(TimeRollingPlotter, self).__init__(ticker, settings,
                                                 data, plot_type)
        # FIXME: currently fits_dict below is an imported module global
        self.fits_dict = fits_dict["time_rolling"]
        # TODO: consider making fits_dict flat in plot_types level

    #  # NOTE: below is WIP
    #  def _get_plot_type_static_info(self):
    #      fig_name = (f"Time rolling {self.curr_ptinfo['display_name']} "
    #                  f"for {self.curr_tdir} tail for {self.curr_ticker}")
    #      self.fig_name = fig_name
