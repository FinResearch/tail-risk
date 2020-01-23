import json


#  Figure Information Templates (FIT) Defined Below

# # Tabled Plots FIT
tabled_figure_fit = {
    "αf":  # α-fitting
    {
        "fig_name": "Alpha Fitting for ${ticker}",
        "vec_types": ("α_vec",),
        "ax_title": ("Time evolution of the parameter "
                     r"$\alpha$ for ${ticker}\n"),
        "ax_ylabel": r"$\alpha$",
        "ax_table":
        {
            "cellText": [(), (), ],
            "cellLoc": "center",
            "colLabels": ("Tail",
                          r"$E[\alpha]$",
                          "Median",
                          r"$\sigma(\alpha)$",
                          "min",
                          "max",),
            "loc": "bottom",
            "bbox": (0.0, -0.26, 1.0, 0.10),
        },
    },
    "hg":  # histogram of tail-α
    {
        "fig_name": "Histogram of ${tail_full_sign} tail alphas for ${ticker}",
        "vec_types": ("α_vec",),
        "extra_lines":
        {
            # NOTE: vectors expr encoded as str; use eval() to get value
            #  "vectors": (("np.repeat(np.mean(data[f'{sign}_α_vec']), np.max(out1) + 1)",
            #               "range(0, np.max(out1) + 1, 1)"),),
            "line_style":
            {
                "label": r"$E[\hat{\alpha}]$",
                "color": "blue",
                "linewidth": 1.5,
            },
        },
        "ax_title": ("Empirical distribution (${tail_dir} tail) of "
                     r"the rolling $\hat{\alpha}$ for ${ticker}\n"),
        "ax_ylabel": "Absolute frequency",
        "ax_table":
        {
            "cellText": "([np.round(fn(${data}[f'${tail_sign}_α_vec']), 4) for fn in (np.mean, np.std, np.min, np.max)], )",
            "cellLoc": "center",
            "colLabels": (r"$E[\hat{\alpha}]$",
                          r"$\sigma (\hat{\alpha})$",
                          "min",
                          "max",),
            "loc": "bottom",
            "bbox": (0.0, -0.26, 1.0, 0.10),
        },
    },
}
# TODO: implement template inheritance for common values


# #  Time Rolling FIT
time_rolling_fit = {
    "ci":  # time rolling confidence interval
    {
        "display_name": "CI",
        "vec_types": ("α_vec", "up_bound", "low_bound"),
        "extra_lines":
        {
            # NOTE: vectors expr encoded as str; use eval() to get value
            "vectors": "map(lambda x: np.repeat(x, ${n_vec} + 2), (2, 3))",
            "line_style": {"color": "red"},
        },
        "ax_title": (r"Rolling confidence intervals for the $\alpha$-"
                     "${tail_dir} tail exponents (c = 1 - ${significance})\n"
                     "Ticker: ${ticker}. "),
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
        "display_name": "size",
        "vec_types": ("abs_len",),
        "ax_title": "Rolling tail length for: ${ticker}\n",
        "ax_ylabel": "Tail length",
    },
    "rs":  # time rolling relative size
    {
        "display_name": "relative size",
        "vec_types": ("rel_len",),
        "ax_title": "Rolling relative tail length for: ${ticker}\n",
        "ax_ylabel": "Relative tail length",
    },
    "ks":  # time rolling KS-test
    {
        "display_name": "KS test",
        "vec_types": ("α_ks",),
        "ax_title": ("KS-statistics: rolling p-value obtained from "
                     "Clauset algorithm for ${ticker}\n"),
        "ax_ylabel": "p-value",
    },
}


fits = {
    "tabled_figure": tabled_figure_fit,
    "time_rolling": time_rolling_fit,
}


for fit_name, tmpl_dict in fits.items():
    with open(f"fit_{fit_name}.json", "w") as fp:
        json.dump(tmpl_dict, fp)
