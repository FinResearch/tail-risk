#  import numpy as np
#  import matplotlib.pyplot as plt

import pylab as z


def plot_boxplot(labels, direction, data, opts, show_plot=False):

    sign = "pos" if direction == "right" else "neg"
    sign_full = "positive" if direction == "right" else "negative"

    z.figure(f"{sign_full.title()} Power Law Boxplot")
    z.boxplot(data[f"{sign}_α_mat"])
    z.xticks(range(1, len(labels) + 1, 1), labels)
    z.xlim(xmin=0.5, xmax=len(labels) + 0.5)
    z.ylabel(r"$\alpha$")
    z.title(
        "Boxplot representation of the "
        + r"$\alpha$"
        + f"-{direction} tail exponent "
        + "\n"
        + "Time Period: "
        + opts.dates[0]
        + " - "
        + opts.dates[-1]
        + ". Input series: "
        #  + lab  # TODO: add this label
    )
    z.grid()

    # NOTE: if "Both" tails, then need to plot both tails on same figure
    if show_plot:
        z.show()
    else:
        # TODO: implement plot saving functionality?
        pass


# Wrapper for plotting the boxplots of alpha tails
def boxplot(labels, data, opts, show_plot=False):

    tails_list = []
    if opts.use_right_tail:
        tails_list.append("right")
    if opts.use_left_tail:
        tails_list.append("left")

    for t in tails_list:
        plot_boxplot(labels, t, data, opts, show_plot=show_plot)
