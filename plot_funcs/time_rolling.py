import itertools
#  import collections

import numpy as np
import matplotlib.pyplot as plt

import pylab as z


# object containing static info; i.e. same for plots of the same type
plot_types_static_info = {
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


class TimeRollingPlotter:

    def __init__(self, data, settings, plot_types_static_info):
        self.data = data
        self.dlens = {k: len(v) for k, v in data}
        self.settings = settings
        self.ptsi = plot_types_static_info

        #  self.plot_types = ("ci", "as", "rs", "ks")
        #  self.ptyp_nmap = {"ci": "CI",
        #                    "as": "size",
        #                    "rs": "relative size",
        #                    "ks": "KS test"}
        #  self.plot2vec_types_map = {"ci": ["α_vec", "up_bound", "low_bound"],
        #                             "as": ["abs_len"],
        #                             "rs": ["rel_len"],
        #                             "ks": ["α_ks"]}
        #  self.vector_types = ("pos_α_vec",     "neg_α_vec",
        #                       "pos_α_ks",      "neg_α_ks",
        #                       "pos_up_bound",  "neg_up_bound",
        #                       "pos_low_bound", "neg_low_bound",
        #                       "pos_abs_len",   "neg_abs_len",
        #                       "pos_rel_len",   "neg_rel_len")

    def _init_plotter_state(self):
        self.curr_ptyp = 'ci'
        self.curr_tail = "right" if settings.use_right_tail else "left"
        self.curr_ticker = self.settings.tickers[0]

    def _update_plotter_state(self):
        pass

    # state-dependent methods below

    def _get_vecs2plot(self):
        """
        TODO: set the correct data to be passed to
              figure_plot & also figure_init
        """

        sett = self.settings
        ptyp = self.curr_ptyp

        vec_prod = itertools.product(["pos", "neg"],  # tail sign: +/-
                                     self.ptsi[ptyp][vec_types])
        vec_names = [f"{sgn}_{typ}" for sgn, typ in vec_prod]

        if right := sett.use_right_tail:
            vec_pos = [vec for vec in vec_names if "pos" in vec]
            vec_neg = []  # NOTE: this is a hack for convenience in final else
        if left := sett.use_left_tail:
            vec_neg = [vec for vec in vec_names if "neg" in vec]

        if right and left:
            if ptyp == "ci":
                vectors = vec_pos, vec_neg   # 2-tuple of lists of str
            else:
                vectors = vec_pos + vec_neg  # single list of str
        else:
            vectors = vec_neg or vec_pos     # single list of str

        # TODO: return same type of "vectors";
        # either tuple or single list, so all downstream API can be unified
        return vectors

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

    def _gen_ax_title(self):

        ptyp = self.curr_ptyp
        ticker = self.curr_ticker
        tail = self.curr_tail

        if ptyp == "ci":
            alpha_sym = r"$\alpha$"  # TODO: consider using α unicode char
            ax_title = ("Rolling confidence intervals for the "
                        f"{alpha_sym}-{tail} tail exponents "
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

    def init_figure(self):
        """Initialize a unique Matplotlib Figure instance,
        to set it up for the actual plotting, then returns it

        TODO: it should not care about the data being plotted nor the opts
        """

        sett = self.settings
        tail = self.curr_tail

        tail_sgn = "pos" if tail == "right" else "neg"

        # TODO: use fig, ax = plt.subplots() idiom to Initialize?
        fig_name = (f"Time rolling {self.ptsi['fig_name']} "
                    f"for {tail} tail for {self.curr_ticker}")
        fig = plt.figure(fig_name)
        axes_pos = (0.1, 0.20, 0.83, 0.70)
        ax = fig.add_axes(axes_pos)

        return ax

    def plot_lines(self, ax):  # , vec_names):
        """Given the data to plot, add plot them onto the passed figure
        TODO: it should not care about tail_dir, opts, and other boilerplate
        """

        vecs2plot = self._get_vecs2plot()

        if self.curr_ptyp == "ci":
            # plot two constant alpha lines in red
            n_vec = len(self.data[vecs2plot[0]])
            # TODO: make n_vec as class attr and use
            const2 = np.repeat(2, n_vec + 2)
            const3 = np.repeat(3, n_vec + 2)
            ax.plot(const2, color="red")  # TODO: make available as class attr
            ax.plot(const3, color="red")  # TODO: make available as class attr

        for vn in vecs2plot:
            ax.plot(self.data[vn], **set_line_style(vn))

    def config_axes(self, ax):
        """
        configure it appropriately after plotting (title, ticks, etc.)
        """

        sett = self.settings
        ptsi = self.ptsi
        ptyp = self.curr_ptyp

        axtit_uniq = self._gen_ax_title()

        axtit_comm = (f"Time Period: {sett.dates[0]} - {sett.dates[-1]}. "
                      f"Input: ")  # + lab  # TODO: add this label
        ax.set_title(axtit_uniq + axtit_comm)

        # TODO: use self.dlens to calculate xmax value
        ax.set_xlim(xmin=0.0, xmax=len(self.data[f"{tail_sgn}_α_vec"]) - 1)
        ax.set_xticks(range(0, len(sett.spec_dates), sett.spec_labelstep))
        ax.set_xticklabels([d[3:] for d in
                            sett.spec_dates[0::sett.spec_labelstep]],
                           rotation="vertical")

        ax.set_ylabel(ptsi[ptyp]["ax_ylabel"])

        ax.legend(**ptsi[ptyp].get("ax_legend", {}))
        #  if "legend" in ax_metainfo:
        #      ax.legend(**ax_metainfo["legend"])
        #  else:
        #      ax.legend()

        ax.grid()

    # NOTE: does this function need to be state aware?
    def present_figure(self, fig):  # , show_plot=False):
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


def plot_ci(ticker, data, opts, show_plot=False):

    #  vecs_pos, vecs_neg = get_vecs2plot("ci", opts)
    vecs2plot_tup = get_vecs2plot("ci", opts)

    for i, vecs_ls in enumerate(vecs2plot_tup):
        # TODO: makw tail_dir into attr associated w/ the vec_list itself
        tail_dir = "right" if i == 0 else "left"

        alpha_sym = r"$\alpha$"  # TODO: consider just using unicode char: α
        ax_title = (f"Rolling confidence intervals for the {alpha_sym}-{tail_dir} "
                    f"tail exponents (c = {1 - opts.significance})\n"
                    f"Ticker: {ticker}. ")

        fig = fig_config(ticker, "CI", tail_dir, data, opts, ax_title)
        fig_plot(fig, "ci", data, vecs_ls)
        fig_present(fig, show_plot)

    #  alpha_sym = r"$\alpha$"  # TODO: consider just using unicode char: α
    #  ax.set_ylabel(alpha_sym)

    #  ax.legend(bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
    #            ncol=3, mode="expand", borderaxespad=0)


def plot_abs_size(ticker, data, opts, show_plot=False):

    vecs2plot = get_vecs2plot("as", opts)

    # TODO: support "right", "left", "both" tails
    tail_dir = "right"

    ax_title = f"Rolling tail length for: {ticker}\n"
    fig = fig_config(ticker, "rolling size", tail_dir, data, opts, ax_title)
    fig_plot(fig, "as", data, vecs2plot)
    fig_present(fig, show_plot)

    #  # NOTE: use inside function for now; factor out later
    #  spec_dates, spec_labelstep = spec_helper(opts)
    #
    #  sign = "pos" if direction == "right" else "neg"
    #  line_color = "green" if direction == "right" else "purple"

    #  z.figure(f"Time rolling size for {direction} tail for {label}")
    #  z.gca().set_position((0.1, 0.20, 0.83, 0.70))

    #  z.plot(data[f"{sign}_abs_len"], color=line_color,
    #         label=f"{direction.title()} tail")

    # TODO: factor out s.t. this fn need not care about which tail(s) selected
    if opts.use_right_tail and opts.use_left_tail:
        opp_dir = "left" if direction == "right" else "left"
        opp_sign = "pos" if opp_dir == "right" else "neg"
        opp_line_color = "green" if opp_dir == "right" else "purple"
        z.plot(data[f"{opp_sign}_abs_len"], color=opp_line_color,
               label=f"{opp_dir.title()} tail")

    z.xticks(
        range(0, len(spec_dates), spec_labelstep),
        [el[3:] for el in spec_dates[0::spec_labelstep]],
        rotation="vertical",
    )


def plot_rel_size(label, direction, data, opts, show_plot=False):

    # NOTE: use inside function for now; factor out later
    spec_dates, spec_labelstep = spec_helper(opts)

    sign = "pos" if direction == "right" else "neg"
    line_color = "green" if direction == "right" else "purple"

    z.figure(f"Time rolling relative size for {direction} tail for {label}")
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    z.plot(data[f"{sign}_rel_len"], color=line_color,
           label=f"{direction.title()} tail")

    # TODO: factor out s.t. this fn need not care about which tail(s) selected
    if opts.use_right_tail and opts.use_left_tail:
        opp_dir = "left" if direction == "right" else "left"
        opp_sign = "pos" if opp_dir == "right" else "neg"
        opp_line_color = "green" if opp_dir == "right" else "purple"
        z.plot(data[f"{opp_sign}_rel_len"], color=opp_line_color,
               label=f"{opp_dir.title()} tail")

    z.ylabel("Relative tail length")
    z.xlim(xmin=0.0, xmax=len(data[f"{sign}_rel_len"]) - 1)
    z.xticks(
        range(0, len(spec_dates), spec_labelstep),
        [el[3:] for el in spec_dates[0::spec_labelstep]],
        rotation="vertical",
    )
    z.title(
        "Rolling relative tail length for :"
        + label
        + "\n"
        + "Time Period: "
        + opts.dates[0]
        + " - "
        + opts.dates[-1]
        + ". Input: "
        #  + lab  # TODO: add this label
    )
    z.legend()
    z.grid()

    # NOTE: if "Both" tails, then need to plot both tails on same figure
    if show_plot:
        z.show()
    else:
        # TODO: implement plot saving functionality?
        pass


def plot_ks_pv(label, direction, data, opts, show_plot=False):

    # NOTE: use inside function for now; factor out later
    spec_dates, spec_labelstep = spec_helper(opts)

    sign = "pos" if direction == "right" else "neg"
    line_color = "green" if direction == "right" else "purple"

    z.figure(f"Time rolling KS test for {direction} tail for {label}")
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    z.plot(data[f"{sign}_α_ks"], color=line_color,
           label=f"{direction.title()} tail")

    # TODO: factor out s.t. this fn need not care about which tail(s) selected
    if opts.use_right_tail and opts.use_left_tail:
        opp_dir = "left" if direction == "right" else "left"
        opp_sign = "pos" if opp_dir == "right" else "neg"
        opp_line_color = "green" if opp_dir == "right" else "purple"
        z.plot(data[f"{opp_sign}_α_ks"], color=opp_line_color,
               label=f"{opp_dir.title()} tail")

    z.ylabel("p-value")
    z.xlim(xmin=0.0, xmax=len(data[f"{sign}_abs_len"]) - 1)
    z.xticks(
        range(0, len(spec_dates), spec_labelstep),
        [el[3:] for el in spec_dates[0::spec_labelstep]],
        rotation="vertical",
    )
    z.title(
        "KS-statistics: rolling p-value obtained from "
        "Clauset algorithm for "
        + label
        + "\n"
        + "Time Period: "
        + opts.dates[0]
        + " - "
        + opts.dates[-1]
        + ". Input: "
        #  + lab  # TODO: add this label
    )
    z.legend()
    z.grid()

    # NOTE: if "Both" tails, then need to plot both tails on same figure
    if show_plot:
        z.show()
    else:
        # TODO: implement plot saving functionality?
        pass


# Plot the alpha exponent confidence interval in time
def time_rolling(label, data, opts, show_plot=False):

    plot_ci(label, data, opts, show_plot=show_plot)

    # NOTE: when "Both" selected, passing either left or right plots both
    #       need to figure out how to plot individual ones
    plot_abs_size(label, data, opts, show_plot=show_plot)

    plot_rel_size(label, "right", data, opts, show_plot=show_plot)
    plot_ks_pv(label, "right", data, opts, show_plot=show_plot)
