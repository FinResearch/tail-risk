import itertools

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
        }
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


# TODO: move this into plotter state generating class later
def _get_all_plot_combos(settings, ptsi):

    tails = []
    if settings.use_right_tail:
        tails.append("right")
    if settings.use_left_tail:
        tails.append("left")

    return tuple(itertools.product(tails, ptsi.keys()))


# TODO: consider making these into plot_types_static_info (ptsi) data?
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

    def __init__(self, ticker, settings, data):  # ptsi, data):
        self.ticker = ticker
        self.settings = settings
        self.ptsi = plot_types_static_info
        #  self.ptsi = ptsi
        self.data = data
        self.dlens = {k: len(v) for k, v in data.items()}  # TODO: assoc to plot_types
        self.all_plot_combos = _get_all_plot_combos(settings, plot_types_static_info)
        #  self.plot_combo_N = len(self.all_plot_combos)

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
        # TODO: above will be diff from self.ticker once plot unnested in ticker-loop

    # state-dependent and aware methods below

    def __get_vecs2plot(self):
        """
        TODO: set the correct data to be passed to
              figure_plot & also figure_init
        """

        sett = self.settings

        vec_prod = itertools.product(["pos", "neg"],  # tail sign: +/-
                                     self.curr_ptsi["vec_types"])
        vec_names = [f"{sgn}_{typ}" for sgn, typ in vec_prod]

        if right := sett.use_right_tail:
            vec_pos = [vec for vec in vec_names if "pos" in vec]
            vec_neg = []  # FIXME: convenience hack for the final else
        if left := sett.use_left_tail:
            vec_neg = [vec for vec in vec_names if "neg" in vec]

        if right and left:
            if self.curr_ptyp == "ci":
                #  vectors = vec_pos, vec_neg   # 2-tuple of lists of str
                return vec_pos, vec_neg   # 2-tuple of lists of str
            else:
                vectors = vec_pos + vec_neg  # single list of str
        else:
            vectors = vec_neg or vec_pos     # single list of str

        # TODO: return same type of "vectors";
        # either tuple or single list, so all downstream API can be unified
        return (vectors,)

    def __gen_ax_title(self):

        ticker = self.curr_ticker
        ptyp = self.curr_ptyp

        if ptyp == "ci":
            alpha_sym = r"$\alpha$"  # TODO: consider using α unicode char
            ax_title = ("Rolling confidence intervals for the "
                        f"{alpha_sym}-{self.curr_tdir} tail exponents "
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

        sett = self.settings

        # TODO: use fig, ax = plt.subplots() idiom to Initialize?
        fig_name = (f"Time rolling {self.curr_ptsi['fig_name']} "
                    f"for {self.curr_tdir} tail for {self.curr_ticker}")
        fig = plt.figure(fig_name)
        axes_pos = (0.1, 0.20, 0.83, 0.70)
        ax = fig.add_axes(axes_pos)

        return ax

    def _plot_lines(self, ax):  # , vec_names):
        """Given the data to plot, add plot them onto the passed figure
        TODO: it should not care about tail_dir, opts, and other boilerplate
        """

        vecs2plot = self.__get_vecs2plot()[0]

        # plot two constant alpha lines in red
        if self.curr_ptyp == "ci":
            # FIXME: with "Both" vecs2plot is a tuple of lists, but when "Right" or
            # "Left" only, it is a list, thus getting the list len() is problematic
            n_vec = len(self.data[vecs2plot[0]])
            # TODO: make n_vec as class attr and use
            const2 = np.repeat(2, n_vec + 2)
            const3 = np.repeat(3, n_vec + 2)
            ax.plot(const2, color="red")  # TODO: make available as class attr
            ax.plot(const3, color="red")  # TODO: make available as class attr

        for vn in vecs2plot:
            # FIXME: same same as FIXME above
            ax.plot(self.data[vn], **_set_line_style(vn))

    def _config_axes(self, ax):
        """
        configure it appropriately after plotting (title, ticks, etc.)
        """

        sett = self.settings

        axtit_uniq = self.__gen_ax_title()
        axtit_comm = (f"Time Period: {sett.dates[0]} - {sett.dates[-1]}. "
                      f"Input: ")  # + lab  # TODO: add this label
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
        """Show or save the plot(s)
        either save individual plots with fig.save,
        or show the plot(s) using plt.show()
        TODO: support interative mode
        # (..., save, interact):
        """

        if self.settings.show_plots:
            plt.show()
        else:
            # TODO: implement plot saving functionality here
            pass

    def plot(self):
        """
        This is the public API
        """

        for tdir, ptyp in self.all_plot_combos:
            self._set_plotter_state(tdir, ptyp)
            ax = self._init_figure()
            self._plot_lines(ax)
            self._config_axes(ax)
            self._present_figure()
