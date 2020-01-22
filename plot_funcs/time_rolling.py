#  import itertools.product as prod
from itertools import product

import numpy as np
import matplotlib.pyplot as plt


# object containing static info; i.e. same for plots of the same type
# TODO: move into separate JSON file in dedicated config directory?
plot_types_static_info = {  # NOTE: will be abbreviated as ptsi
    "ci":
    {
        "fig_name": "CI",
        "vec_types": ("α_vec", "up_bound", "low_bound"),
        "ax_ylabel": r"$\alpha$",
        "ax_legend":
        {
            "bbox_to_anchor": (0.0, -0.175, 1.0, 0.02),
            "ncol": 3,
            "mode": "expand",
            "borderaxespad": 0
        },
    },
    "as":
    {
        "fig_name": "size",
        "vec_types": ("abs_len",),
        "ax_ylabel": "Tail length",
    },
    "rs":
    {
        "fig_name": "relative size",
        "vec_types": ("rel_len",),
        "ax_ylabel": "Relative tail length",
    },
    "ks":
    {
        "fig_name": "KS test",
        "vec_types": ("α_ks",),
        "ax_ylabel": "p-value",
    },
}


# TODO: consider making values returned from this function
#       part of plot_types_static_info (ptsi) data
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
class TimeRollingPlotter:
    """
    Note on method name conventions: other than the reserved dunder methods,
    self-defined methods prepended by a double underscore are meant to be
    called only by other private methods, which are themselves prepended by
    a single underscore.
    """

    def __init__(self, ticker, settings, data):  # ptsi, data):
        self.ticker = ticker
        self.settings = settings
        #  self.ptsi = ptsi  # TODO: pass this as a config object
        self.ptsi = plot_types_static_info
        self.data = data
        # TODO: assoc vec_length attr below to plot_types (ptsi?)
        #  self.dlens = {k: len(v) for k, v in data.items()}
        self.tails_used = self.__get_tails_used()
        self.all_plot_combos = self.__get_all_plot_combos()
        self.return_type_label = self.__get_return_type_label()

    # Methods for determining state-independent info (called in __init__)

    def __get_tails_used(self):
        """Return tuple containing the tails selected/used
        """

        tails_used = []
        if self.settings.use_right_tail:
            tails_used.append("right")
        if self.settings.use_left_tail:
            tails_used.append("left")

        return tuple(tails_used)

    def __get_all_plot_combos(self):
        """Return tuple of 2-tups representing all concrete figures requested
        """
        # TODO: i.e. when plt_typ is one of ["as", "rs", "ks"],
        # then need to also do a combined fig of pos+neg tails
        return tuple(product(self.tails_used, self.ptsi.keys()))

    def __get_return_type_label(self):

        pi = "P(t)"
        pf = f"P(t+{self.settings.tau})"

        if self.settings.return_type == "basic":
            label = f"{pf} - {pi}"
        elif self.settings.return_type == "relative":
            label = f"{pf}/{pi} - 1.0"
        elif self.settings.return_type == "log":
            label = rf"$\log$({pf}/{pi})"

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
        self.curr_tsgn = "pos" if self.curr_tdir == "right" else "neg"
        self.curr_ptsi = self.ptsi[self.curr_ptyp]
        #  self.curr_ticker = self.settings.tickers[0]
        self.curr_ticker = self.ticker  # TODO: will be diff when plot unnested
        # TODO: above will be diff from self.ticker once unnested in tickers

    # State-aware and -dependent methods below

    def __get_vecs2plot(self):
        """
        Set the correct data to be passed to _plot_lines()
        """
        # TODO: use this function to do processing on vecs2plot?
        # For example, when plt_typ is one of ["as", "rs", "ks"],
        # then do a combined plot of pos + neg

        # TODO: consider making curr_tsgn into a tuple instead of just a str
        # so that ("right",) or ("right", "left") can be producted w/ vec_types
        return [f"{self.curr_tsgn}_{ptyp}" for ptyp
                in self.curr_ptsi["vec_types"]]

    def __gen_ax_title(self):

        ticker = self.curr_ticker
        ptyp = self.curr_ptyp

        # TODO: consider moving these partial titles into some config obj/file
        if ptyp == "ci":
            #  alpha_tex = r"$\alpha$"
            # TODO: consider using 'α' unicode char instead of TeX markup
            ax_title = ("Rolling confidence intervals for the "
                        rf"$\alpha$-{self.curr_tdir} tail exponents "
                        f"(c = {1 - self.settings.significance})"
                        f"\nTicker: {ticker}. ")
        elif ptyp == "as":
            ax_title = f"Rolling tail length for: {ticker}\n"
        elif ptyp == "rs":
            ax_title = f"Rolling relative tail length for: {ticker}\n"
        elif ptyp == "ks":
            ax_title = ("KS-statistics: rolling p-value obtained from "
                        f"Clauset algorithm for {ticker}\n")

        return ax_title

    def _init_figure(self):
        """Initialize a unique Matplotlib Figure instance,
        to set it up for the actual plotting, then returns it

        TODO: it should not care about the data being plotted nor the opts
        """

        # TODO: use fig, ax = plt.subplots() idiom to Initialize?
        fig_name = (f"Time rolling {self.curr_ptsi['fig_name']} "
                    f"for {self.curr_tdir} tail for {self.curr_ticker}")
        fig = plt.figure(fig_name)
        axes_pos = (0.1, 0.20, 0.83, 0.70)
        ax = fig.add_axes(axes_pos)

        return ax

    def _plot_lines(self, ax):  # , vec_names):
        """Given the data to plot, add plot them onto the passed figure
        """
        # TODO: it should not care about tail_dir, opts, and other boilerplate

        vecs2plot = self.__get_vecs2plot()

        # plot two constant alpha lines in red
        if self.curr_ptyp == "ci":
            # TODO: make n_vec as class attr and use
            n_vec = len(self.data[vecs2plot[0]])
            const2 = np.repeat(2, n_vec + 2)
            const3 = np.repeat(3, n_vec + 2)
            ax.plot(const2, color="red")  # TODO: make available as class attr
            ax.plot(const3, color="red")  # TODO: make available as class attr

        for vn in vecs2plot:
            ax.plot(self.data[vn], **_set_line_style(vn))

    def _config_axes(self, ax):
        """
        configure it appropriately after plotting (title, ticks, etc.)
        """

        sett = self.settings

        axtit_uniq = self.__gen_ax_title()
        axtit_comm = (f"Time Period: {sett.date_i} - {sett.date_f}. "
                      f"Input: {self.return_type_label}")
        ax.set_title(axtit_uniq + axtit_comm)

        # TODO: use self.dlens to calculate xmax value
        ax.set_xlim(xmin=0.0, xmax=len(self.data[f"{self.curr_tsgn}_α_vec"]) - 1)
        ax.set_xticks(range(0, len(sett.spec_dates), sett.spec_labelstep))
        ax.set_xticklabels([d[3:] for d in
                            sett.spec_dates[0::sett.spec_labelstep]],
                           rotation="vertical")

        ax.set_ylabel(self.curr_ptsi["ax_ylabel"])

        ax.legend(**self.curr_ptsi.get("ax_legend", {}))
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
