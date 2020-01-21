import numpy as np
import matplotlib.pyplot as plt

import pylab as z


# TODO: extract into own file to share with other plot_funcs
def spec_helper(opts):
    if opts.analysis_freq > 1:
        spec_dates = []
        for ddd in range(0, len(opts.dates), opts.analysis_freq):
            spec_dates.append(opts.dates[ddd])
        spec_labelstep = 22
    else:
        spec_dates = opts.dates
        spec_labelstep = opts.labelstep
    return spec_dates, spec_labelstep


def fig_config(ticker, plot_type, opts):
    # TODO: set the correct data to be passed to figure_plot & also figure_init
    pass


def fig_make(ticker, plt_type, tail_dir, data, opts):
    """Initialize a unique Matplotlib Figure instance, and returns it
    TODO: it should not care about the data being plotted nor the opts
    """

    # NOTE: use inside function for now; factor out later -> into opts?
    spec_dates, spec_labelstep = spec_helper(opts)

    tail_sgn = "pos" if tail_dir == "right" else "neg"

    # TODO: use fig, ax = plt.subplots() idiom to Initialize

    fig_name = f"Time rolling {plt_type} for {tail_dir} tail for {ticker}"
    fig = plt.figure(fig_name)

    axes_pos = (0.1, 0.20, 0.83, 0.70)
    ax = fig.add_axes(axes_pos)

    alpha_sym = r"$\alpha$"  # TODO: consider just using unicode char: α
    ax_title = (f"Rolling confidence intervals for the {alpha_sym}-{tail_dir} "
                f"tail exponents (c = {1 - opts.significance})\n"
                f"Ticker: {ticker}. "
                f"Time Period: {opts.dates[0]} - {opts.dates[-1]}. "
                f"Input: ")  # + lab  # TODO: add this label
    ax.set_title(ax_title)

    # TODO: rid dependency on data arg in xmax below
    ax.set_xlim(xmin=0.0, xmax=len(data[f"{tail_sgn}_α_vec"]) - 1)
    ax.set_xticks(range(0, len(spec_dates), spec_labelstep))  # must be list?
    ax.set_xticklabels([el[3:] for el in spec_dates[0::spec_labelstep]],
                       rotation="vertical")

    ax.set_ylabel(alpha_sym)

    #  ax.legend(bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
    #            ncol=3, mode="expand", borderaxespad=0)
    #  ax.grid()

    return fig


def fig_plot(fig, tail_dir, data):
    """Given the data to plot, add plot them onto the passed figure
    TODO: it should not care about tail_dir, opts, and other boilerplate
    """

    tail_sgn = "pos" if tail_dir == "right" else "neg"

    ax = fig.get_axes()[0]  # TODO: pass ax as param to avoid list-index
    ax.plot(data[f"{tail_sgn}_α_vec"], color="green",
            label=f"{tail_dir.title()} tail")
    ax.plot(data[f"{tail_sgn}_up_bound"], color="purple", label="Upper bound")
    ax.plot(data[f"{tail_sgn}_low_bound"], color="blue", label="Lower bound")
    ax.plot(np.repeat(3, len(data[f"{tail_sgn}_α_vec"]) + 2), color="red")
    ax.plot(np.repeat(2, len(data[f"{tail_sgn}_α_vec"]) + 2), color="red")


def fig_present(fig, show_plot):  # , save, interact):
    """Show or save the plot(s)
    either save individual plots with fig.save,
    or show the plot(s) using plt.show()
    TODO: support interative mode
    """

    if show_plot:
        plt.show()
    else:
        # TODO: implement plot saving functionality?
        pass


def plot_ci(ticker, tail_dir, data, opts, show_plot=False):

    fig = fig_make(ticker, "CI", tail_dir, data, opts)
    fig_plot(fig, tail_dir, data)
    fig_present(fig, show_plot)

    #  z.legend(
    #      bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
    #      ncol=3,
    #      mode="expand",
    #      borderaxespad=0,
    #  )
    #  z.grid()


def plot_abs_size(label, direction, data, opts, show_plot=False):

    # NOTE: use inside function for now; factor out later
    spec_dates, spec_labelstep = spec_helper(opts)

    sign = "pos" if direction == "right" else "neg"
    line_color = "green" if direction == "right" else "purple"

    z.figure(f"Time rolling size for {direction} tail for {label}")
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    z.plot(data[f"{sign}_abs_len"], color=line_color,
           label=f"{direction.title()} tail")

    # TODO: factor out s.t. this fn need not care about which tail(s) selected
    if opts.use_right_tail and opts.use_left_tail:
        opp_dir = "left" if direction == "right" else "left"
        opp_sign = "pos" if opp_dir == "right" else "neg"
        opp_line_color = "green" if opp_dir == "right" else "purple"
        z.plot(data[f"{opp_sign}_abs_len"], color=opp_line_color,
               label=f"{opp_dir.title()} tail")

    z.ylabel("Tail length")
    z.xlim(xmin=0.0, xmax=len(data[f"{sign}_abs_len"]) - 1)
    z.xticks(
        range(0, len(spec_dates), spec_labelstep),
        [el[3:] for el in spec_dates[0::spec_labelstep]],
        rotation="vertical",
    )
    z.title(
        "Rolling tail length for :"
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

    tails_list = []
    if opts.use_right_tail:
        tails_list.append("right")
    if opts.use_left_tail:
        tails_list.append("left")

    for t in tails_list:
        plot_ci(label, t, data, opts, show_plot=show_plot)

    # NOTE: when "Both" selected, passing either left or right plots both
    #       need to figure out how to plot individual ones
    plot_abs_size(label, "right", data, opts, show_plot=show_plot)
    plot_rel_size(label, "right", data, opts, show_plot=show_plot)
    plot_ks_pv(label, "right", data, opts, show_plot=show_plot)
