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


# Plot the alpha exponent confidence interval in time
def time_rolling(label, data, opts, show_plot=False):

    tails_list = []
    if opts.use_right_tail:
        tails_list.append("right")
    if opts.use_left_tail:
        tails_list.append("left")

    for t in tails_list:
        plot_ci(label, t, data, opts, show_plot=show_plot)


#  def hurr():
#
#      #  if tail_selected == "Both" or tail_selected == "Right":
#
#      z.figure("Time rolling size for right tail for " + labels[i - 1])
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      z.plot(results["pos_abs_len"], color="green", label="Right tail")
#      if tail_selected == "Both":
#          z.plot(results["neg_abs_len"],
#                 color="purple", label="Left tail")
#      z.ylabel("Tail length")
#      z.xlim(xmin=0.0, xmax=len(results["pos_abs_len"]) - 1)
#      z.xticks(
#          range(0, len(spec_dates), spec_labelstep),
#          [el[3:] for el in spec_dates[0::spec_labelstep]],
#          rotation="vertical",
#      )
#      z.title(
#          "Rolling tail length for :"
#          + labels[i - 1]
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input: "
#          + lab
#      )
#      z.legend()
#      z.grid()
#      # z.show()
#
#      z.figure("Time rolling relative size for right tail for "
#               + labels[i - 1])
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      z.plot(results["pos_rel_len"], color="green", label="Right tail")
#      if tail_selected == "Both":
#          z.plot(results["neg_rel_len"],
#                 color="purple", label="Left tail")
#      z.ylabel("Relative tail length")
#      z.xlim(xmin=0.0, xmax=len(results["pos_rel_len"]) - 1)
#      z.xticks(
#          range(0, len(spec_dates), spec_labelstep),
#          [el[3:] for el in spec_dates[0::spec_labelstep]],
#          rotation="vertical",
#      )
#      z.title(
#          "Rolling relative tail length for :"
#          + labels[i - 1]
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input: "
#          + lab
#      )
#      z.legend()
#      z.grid()
#      # z.show()
#
#      z.figure("Time rolling KS test for right tail for " + labels[i-1])
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      z.plot(results["pos_α_ks"], color="green", label="Right tail")
#      if tail_selected == "Both":
#          z.plot(results["neg_α_ks"], color="purple", label="Left tail")
#      z.ylabel("p-value")
#      z.xlim(xmin=0.0, xmax=len(results["pos_abs_len"]) - 1)
#      z.xticks(
#          range(0, len(spec_dates), spec_labelstep),
#          [el[3:] for el in spec_dates[0::spec_labelstep]],
#          rotation="vertical",
#      )
#      z.title(
#          "KS-statistics: rolling p-value obtained from "
#          "Clauset algorithm for "
#          + labels[i - 1]
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input: "
#          + lab
#      )
#      z.legend()
#      z.grid()
#      # z.show()
#
#      #  if tail_selected == "Both" or tail_selected == "Left":
#
#      z.figure("Time rolling size for left tail for " + labels[i - 1])
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      z.plot(results["neg_abs_len"], color="purple", label="Left tail")
#      if tail_selected == "Both":
#          z.plot(results["pos_abs_len"],
#                 color="green", label="Right tail")
#      z.ylabel("Tail length")
#      z.xlim(xmin=0.0, xmax=len(results["neg_abs_len"]) - 1)
#      z.xticks(
#          range(0, len(spec_dates), spec_labelstep),
#          [el[3:] for el in spec_dates[0::spec_labelstep]],
#          rotation="vertical",
#      )
#      z.title(
#          "Rolling tail length for :"
#          + labels[i - 1]
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input: "
#          + lab
#      )
#      z.legend()
#      z.grid()
#      # z.show()
#
#      z.figure("Time rolling relative size for left tail for " +
#               labels[i - 1])
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      z.plot(results["neg_rel_len"], color="purple", label="Left tail")
#      if tail_selected == "Both":
#          z.plot(results["pos_rel_len"],
#                 color="green", label="Right tail")
#      z.ylabel("Relative tail length")
#      z.xlim(xmin=0.0, xmax=len(results["neg_rel_len"]) - 1)
#      z.xticks(
#          range(0, len(spec_dates), spec_labelstep),
#          [el[3:] for el in spec_dates[0::spec_labelstep]],
#          rotation="vertical",
#      )
#      z.title(
#          "Rolling relative tail length for :"
#          + labels[i - 1]
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input: "
#          + lab
#      )
#      z.legend()
#      z.grid()
#      # z.show()
#
#      z.figure("Time rolling KS test for left tail for " + labels[i - 1])
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      z.plot(results["neg_α_ks"], color="purple", label="Left tail")
#      if tail_selected == "Both":
#          z.plot(results["pos_α_ks"], color="green", label="Right tail")
#      z.ylabel("p-value")
#      z.xlim(xmin=0.0, xmax=len(results["neg_abs_len"]) - 1)
#      z.xticks(
#          range(0, len(spec_dates), spec_labelstep),
#          [el[3:] for el in spec_dates[0::spec_labelstep]],
#          rotation="vertical",
#      )
#      z.title(
#          "KS-statistics: rolling p-value obtained "
#          "from Clauset algorithm for "
#          + labels[i - 1]
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input: "
#          + lab
#      )
#      z.legend()
#      z.grid()
#      # z.show()
