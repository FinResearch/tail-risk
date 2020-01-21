import itertools

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


def get_vecs2plot(plot_type, opts):
    """
    TODO: set the correct data to be passed to figure_plot & also figure_init
    """

    plt_typ2vec = {"ci": ["α_vec", "up_bound", "low_bound"],  # conf interval
                   "as": ["abs_len"],                         # absolute size
                   "rs": ["rel_len"],                         # relative size
                   "ks": ["α_ks"]}                            # KS-test
    vec_prod = itertools.product(["pos", "neg"], plt_typ2vec[f"{plot_type}"])
    vec_names = [f"{sgn}_{typ}" for sgn, typ in vec_prod]

    if right := opts.use_right_tail:
        data_pos = [vec for vec in vec_names if "pos" in vec]
        data_neg = []  # NOTE: this is a hack to make final else convenient

    if left := opts.use_left_tail:
        data_neg = [vec for vec in vec_names if "neg" in vec]

    if right and left:
        if plot_type == "ci":
            data = data_pos, data_neg   # 2-tuple of lists of str
        else:   # plot_type is one of {absolute size, relative size, KS-test}
            data = data_pos + data_neg  # single list of str (combined tails)
    else:
        data = data_neg or data_pos     # single list of str

    # TODO: return same data type;
    # either tuple or single list, so all downstream API can be unified
    return data


def set_line_style(vec_name):
    """Helper for setting the line style of the line plot
    :param: vec_name: string name of the vector to be plotted
    """

    if "pos" in vec_name:
        label = "Right tail"
        color = "green"
    elif "neg" in vec_name:
        label = "Left tail"
        color = "purple"

    # overwrite color and line if plotting alpha boundary curves
    if "up_bound" in vec_name:
        label = "Upper bound"
        color = "black"
    elif "low_bound" in vec_name:
        label = "Lower bound"
        color = "blue"

    return {"label": label, "color": color}


def fig_config(ticker, plot_type, tail_dir, data, opts):
    """Initialize a unique Matplotlib Figure instance,
    configure it appropriately (title, ticks, etc.),
    to set it up for the actual plotting, then returns it

    TODO: it should not care about the data being plotted nor the opts
    """

    # NOTE: use inside function for now; factor out later -> into opts?
    spec_dates, spec_labelstep = spec_helper(opts)

    tail_sgn = "pos" if tail_dir == "right" else "neg"

    # TODO: use fig, ax = plt.subplots() idiom to Initialize

    fig_name = f"Time rolling {plot_type} for {tail_dir} tail for {ticker}"
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

    ax.legend(bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
              ncol=3, mode="expand", borderaxespad=0)
    ax.grid()

    return fig


def fig_plot(fig, plot_type, data, vec_names):
    """Given the data to plot, add plot them onto the passed figure
    TODO: it should not care about tail_dir, opts, and other boilerplate
    """

    ax = fig.get_axes()[0]  # TODO: pass ax as param to avoid list-indexing

    if plot_type == "ci":
        # plot two constant alpha lines in red
        n_vec = len(data[vec_names[0]])  # TODO: make available as class attr
        const2 = np.repeat(2, n_vec + 2)
        const3 = np.repeat(3, n_vec + 2)
        ax.plot(const2, color="red")     # TODO: make available as class attr
        ax.plot(const3, color="red")     # TODO: make available as class attr

    # TODO: rid dependency on data var below; make data a class attr
    for vn in vec_names:
        ax.plot(data[vn], **set_line_style(vn))


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


def plot_ci(ticker, data, opts, show_plot=False):

    #  vecs_pos, vecs_neg = get_vecs2plot("ci", opts)
    vecs2plot_tup = get_vecs2plot("ci", opts)

    for i, vecs_ls in enumerate(vecs2plot_tup):
        # TODO: makw tail_dir into attr associated w/ the vec_list itself
        tail_dir = "right" if i == 0 else "left"
        fig = fig_config(ticker, "CI", tail_dir, data, opts)
        fig_plot(fig, "ci", data, vecs_ls)
        fig_present(fig, show_plot)



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

    plot_ci(label, data, opts, show_plot=show_plot)

    # NOTE: when "Both" selected, passing either left or right plots both
    #       need to figure out how to plot individual ones
    plot_abs_size(label, "right", data, opts, show_plot=show_plot)
    plot_rel_size(label, "right", data, opts, show_plot=show_plot)
    plot_ks_pv(label, "right", data, opts, show_plot=show_plot)
