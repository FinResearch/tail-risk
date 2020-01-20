import numpy as np
#  import matplotlib.pyplot as plt

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


def plot_ci(label, direction, data, opts, show_plot=False):

    # NOTE: use inside function for now; factor out later
    spec_dates, spec_labelstep = spec_helper(opts)

    sign = "pos" if direction == "right" else "neg"

    z.figure("Time rolling CI for " + direction + " tail for " + label)
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    z.plot(data[f"{sign}_α_vec"], color="green",
           label=f"{direction.title()} tail")
    z.plot(data[f"{sign}_up_bound"], color="purple", label="Upper bound")
    z.plot(data[f"{sign}_low_bound"], color="blue", label="Lower bound")
    z.plot(np.repeat(3, len(data[f"{sign}_α_vec"]) + 2), color="red")
    z.plot(np.repeat(2, len(data[f"{sign}_α_vec"]) + 2), color="red")
    z.ylabel(r"$\alpha$")
    z.xlim(xmin=0.0, xmax=len(data[f"{sign}_α_vec"]) - 1)
    z.xticks(
        range(0, len(spec_dates), spec_labelstep),
        [el[3:] for el in spec_dates[0::spec_labelstep]],
        rotation="vertical",
    )
    z.title(
        "Rolling confidence intervals for the "
        + r"$\alpha$"
        + "-"
        + direction
        + " tail exponents "
        + "(c = "
        + str(1 - opts.significance)
        + ")"
        + "\n"
        + "Ticker: "
        + label
        + ".Time Period: "  # ASK: period should be there?
        + opts.dates[0]
        + " - "
        + opts.dates[-1]
        + ". Input: "
        #  + lab  # TODO: add this label
    )
    z.legend(
        bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
        ncol=3,
        mode="expand",
        borderaxespad=0,
    )
    z.grid()

    if show_plot:
        z.show()
    else:
        # TODO: implement plot saving functionality?
        pass


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
