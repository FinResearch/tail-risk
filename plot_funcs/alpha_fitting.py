import numpy as np
import matplotlib.pyplot as plt

import pylab as z


# Plot the alpha exponent in time (right/left/both tail)
def alpha_fitting(label, data, opts, show_plot=False):
    """
    :param: label: string of ticker name
    :param: data: dictionary of lists/arrays containing data to plot
    :param: opts: SimpleNamespace object containing user-input options
    :param: spec_dates: list of dates?
    """

    z.figure("Alpha Fitting for " + label)
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))

    if opts.use_right_tail:
        z.plot(data["pos_α_vec"], label="Right tail")
        z.xlim(xmin=0.0, xmax=len(data["pos_α_vec"]) - 1)
    if opts.use_left_tail:
        z.plot(data["neg_α_vec"], label="Left tail")
        z.xlim(xmin=0.0, xmax=len(data["neg_α_vec"]) - 1)

    z.ylabel(r"$\alpha$")
    z.title(
        "Time evolution of the parameter "
        + r"$\alpha$"
        + " for "
        + label
        + "\n"
        + "Time period: "
        + opts.dates[0]
        + " - "
        + opts.dates[-1]
    )

    # TODO: factor/optimize this out/away
    if opts.analysis_freq > 1:
        spec_dates = []
        for ddd in range(0, len(opts.dates), opts.analysis_freq):
            spec_dates.append(opts.dates[ddd])
        spec_labelstep = 22
    else:
        spec_dates = opts.dates
        spec_labelstep = opts.labelstep

    z.xticks(
        range(0, len(spec_dates), spec_labelstep),
        [el[3:] for el in spec_dates[0::spec_labelstep]],
        rotation="vertical",
    )
    z.grid()
    z.legend()

    # A table with the four statistical moments is built
    col_labels = [
        "Tail",
        r"$E[\alpha]$",
        "Median",
        r"$\sigma(\alpha)$",
        "min",
        "max",
    ]

    table_vals = []
    if opts.use_right_tail:
        # tail_selected == "Right" or tail_selected == "Both":
        table_vals.append(
            [
                "Right",
                np.round(np.mean(data["pos_α_vec"]), 4),
                np.round(np.median(data["pos_α_vec"]), 4),
                np.round(np.std(data["pos_α_vec"]), 4),
                np.round(np.min(data["pos_α_vec"]), 4),
                np.round(np.max(data["pos_α_vec"]), 4),
            ]
        )
    if opts.use_left_tail:
        # tail_selected == "Left" or tail_selected == "Both":
        table_vals.append(
            [
                "Left",
                np.round(np.mean(data["neg_α_vec"]), 4),
                np.round(np.median(data["neg_α_vec"]), 4),
                np.round(np.std(data["neg_α_vec"]), 4),
                np.round(np.min(data["neg_α_vec"]), 4),
                np.round(np.max(data["neg_α_vec"]), 4),
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

    if show_plot:
        z.show()
