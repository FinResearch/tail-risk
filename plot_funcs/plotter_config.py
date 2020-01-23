import json


#  Figure Information Templates (FIT) Defined Below

# Tabled Plots FIT
tabled_plot_fit = {
    "αf":  # α-fitting
    {
        "fig_name": "Alpha Fitting for ${curr_ticker}",
        "vec_types": ("α_vec",),
        "ax_ylabel": r"$\alpha$",
        "ax_table":
        {
            "bbox": (0.0, -0.26, 1.0, 0.10),
            "cellLoc": "center",
            "loc": "bottom",
        },
    },
    "hg":  # histogram of tail-α
    {
        "fig_name": "Histogram of ${curr_tdir} tail alphas for ${curr_ticker}",
        "vec_types": ("α_vec",),
        "extra_lines":
        {
            # NOTE: vectors expr encoded as str; use eval() to get value
            "vectors": (("np.repeat(np.mean(data[f'{sign}_α_vec']), np.max(out1) + 1)",
                         "range(0, np.max(out1) + 1, 1)"),),
            "line_style":
            {
                "label": r"$E[\hat{\alpha}]$",
                "color": "blue",
                "linewidth": 1.5,
             },
        },
        "ax_ylabel": "Absolute frequency",
        "ax_table":
        {
            "bbox": (0.0, -0.26, 1.0, 0.10),
            "cellLoc": "center",
            "loc": "bottom",
        },
    },
}


#  Time Rolling FIT
time_rolling_fit = {
    "ci":  # time rolling confidence interval
    {
        "fig_name": "CI",
        "vec_types": ("α_vec", "up_bound", "low_bound"),
        "extra_lines":
        {
            # NOTE: vectors expr encoded as str; use eval() to get value
            "vectors": "map(lambda x: np.repeat(x, ${n_vec} + 2), (2, 3))",
            "line_style": {"color": "red"},
        },
        "ax_ylabel": r"$\alpha$",
        "ax_legend":
        {
            "bbox_to_anchor": (0.0, -0.175, 1.0, 0.02),
            "ncol": 3,
            "mode": "expand",
            "borderaxespad": 0
        },
    },
    "as":  # time rolling absolute size
    {
        "fig_name": "size",
        "vec_types": ("abs_len",),
        "ax_ylabel": "Tail length",
    },
    "rs":  # time rolling relative size
    {
        "fig_name": "relative size",
        "vec_types": ("rel_len",),
        "ax_ylabel": "Relative tail length",
    },
    "ks":  # time rolling KS-test
    {
        "fig_name": "KS test",
        "vec_types": ("α_ks",),
        "ax_ylabel": "p-value",
    },
}


fits = {
    "tabled_plot": tabled_plot_fit,
    "time_rolling": time_rolling_fit,
}


for fit_name, tmpl_dict in fits.items():
    with open(f"fit_{fit_name}.json", "w") as fp:
        json.dump(tmpl_dict, fp)
