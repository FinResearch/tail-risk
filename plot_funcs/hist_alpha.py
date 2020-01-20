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


def plot_hist(label, direction, data, opts, show_plot=False):

    # NOTE: use inside function for now; factor out later
    spec_dates, spec_labelstep = spec_helper(opts)

    sign = "pos" if direction == "right" else "neg"
    sign_full = "positive" if direction == "right" else "negative"

    z.figure(f"Histogram of {sign_full} tail alphas for {label}",
             figsize=(8, 6), dpi=100)
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))

    IQR = (np.percentile(data[f"{sign}_α_vec"], 75) -
           np.percentile(data[f"{sign}_α_vec"], 25))
    h = 2 * IQR * np.power(len(data[f"{sign}_α_vec"]), -1.0 / 3.0)
    nbins = np.int((np.max(data[f"{sign}_α_vec"]) -
                    np.min(data[f"{sign}_α_vec"])) / float(h))

    # Building the histogram and plotting the relevant vertical lines
    z.hist(data[f"{sign}_α_vec"], nbins, color="red")
    out1, bins = z.histogram(data[f"{sign}_α_vec"], nbins)

    z.plot(
        np.repeat(np.mean(data[f"{sign}_α_vec"]), np.max(out1) + 1),
        range(0, np.max(out1) + 1, 1),
        color="blue",
        linewidth=1.5,
        label=r"$E[\hat{\alpha}]$",
    )

    # Adding the labels, the axis limits, legend and grid
    #  z.xlabel(lab)  # TODO: add label var to opts dict
    z.ylabel("Absolute frequency")
    z.title(
        f"Empirical distribution ({direction} tail) of the rolling "
        + r"$\hat{\alpha}$"
        + " for "
        + label
        + "\n"
        + "Time period: "
        + opts.dates[0]
        + " - "
        + opts.dates[-1]
    )
    z.xlim(xmin=np.min(data[f"{sign}_α_vec"]),
           xmax=np.max(data[f"{sign}_α_vec"]))
    z.ylim(ymin=0, ymax=np.max(out1))
    z.legend()
    z.grid()
    # A table with the four statistical moments is built
    col_labels = [
        r"$E[\hat{\alpha}]$",
        r"$\sigma (\hat{\alpha})$",
        "min",
        "max",
    ]
    table_vals = []
    table_vals.append(
        [
            np.round(np.mean(data[f"{sign}_α_vec"]), 4),
            np.round(np.std(data[f"{sign}_α_vec"]), 4),
            np.round(np.min(data[f"{sign}_α_vec"]), 4),
            np.round(np.max(data[f"{sign}_α_vec"]), 4),
        ]
    )
    the_table = plt.table(
        cellText=table_vals,
        cellLoc="center",
        colLabels=col_labels,
        loc="bottom",
        bbox=[0.0, -0.26, 1.0, 0.10],
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(0.5, 0.5)

    # NOTE: if "Both" tails, then need to plot both tails on same figure
    if show_plot:
        z.show()
    else:
        # TODO: implement plot saving functionality?
        pass


# Plotting the histograms for the rolling alpha
def hist_alpha(label, data, opts, show_plot=False):

    tails_list = []
    if opts.use_right_tail:
        tails_list.append("right")
    if opts.use_left_tail:
        tails_list.append("left")

    for t in tails_list:
        plot_hist(label, t, data, opts, show_plot=show_plot)
