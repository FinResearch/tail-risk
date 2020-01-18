#####################################
# Libraries                         #
#####################################
#  import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as z

import scipy.stats as st
import powerlaw as pl
#  import easygui as eg

import plpva as plpva

#####################################
# Tools Functions                   #
#####################################


def Extractor(filename, tickers):
    object = pd.read_csv(filename)
    output = [(object["Date"].values).tolist()]
    for i in range(0, len(tickers), 1):
        try:
            output.append((object[tickers[i]].values).tolist())
        except KeyError:
            print("Ticker " + tickers[i] + " not found in " + filename)
    return output


def PowerLawFit(data, data_nature, xmin_rule, *args):

    data1 = np.array(data)[np.nonzero(data)]
    discrete = False if data_nature == 'Continuous' else True

    if xmin_rule == "Clauset":
        xmin = None
    elif xmin_rule == "Manual":
        xmin = args[0]
    else:
        xmin = np.percentile(data, args[1])

    return pl.Fit(data1, discrete=discrete, xmin=xmin)


#####################################
# Script inputs (hardcoded for the time)                     #
#####################################

database_name = "dbMSTR_test.csv"

no_entries = 1
fieldNames = ["# " + str(i) for i in range(1, no_entries + 1, 1)]
fieldValues = ["DE 01Y"]  # , "DE 03Y", "DE 05Y", "DE 10Y"]
labels = fieldValues

database = Extractor(database_name, labels)

initial_date = "1/1/2016"
final_date = "5/5/2016"
lookback = 504

input_type = "Log-Returns"

tau = 1

standardize = "No"
abs_value = "No"

approach = "Rolling"
an_freq = 1
tail_selected = "Both"
data_nature = "Continuous"

xmin_rule = "Clauset"

if tail_selected == "Both":
    multiplier = 0.5
else:
    multiplier = 1.0

significance = 0.05

c_iter = 100


def results_lists_init():
    labels = ("pos_α_vec", "neg_α_vec", "pos_α_ks", "neg_α_ks",
              "pos_up_bound", "neg_up_bound", "pos_low_bound", "neg_low_bound",
              "pos_abs_len", "neg_abs_len", "pos_rel_len", "neg_rel_len",
              "loglr_right", "loglr_left", "loglpv_right", "loglpv_left")
    return {l: [] for l in labels}


# initialize lists used to store results (16 total)
results = results_lists_init()


if approach == "Static":

    #  positive_alpha_vec = []
    #  negative_alpha_vec = []
    #  positive_alpha_KS = []
    #  negative_alpha_KS = []
    #  positive_upper_bound = []
    #  positive_lower_bound = []
    #  negative_upper_bound = []
    #  negative_lower_bound = []
    #  positive_abs_length = []
    #  positive_rel_length = []
    #  negative_abs_length = []
    #  negative_rel_length = []

    initial_index = database[0].index(initial_date)
    final_index = database[0].index(final_date)
    dates = database[0][initial_index: (final_index + 1)]
    labelstep = (
        22
        if len(dates) <= 252
        else 66
        if (len(dates) > 252 and len(dates) <= 756)
        else 121
    )
    N = len(database)

    # TODO: add list below to results_lists_init function
    tail_statistics = []

    for i in range(1, N, 1):

        #  loglikelihood_ratio_right = []
        #  loglikelihood_pvalue_right = []
        #  loglikelihood_ratio_left = []
        #  loglikelihood_pvalue_left = []

        print("I am analyzing the time series for " +
              labels[i - 1] + " between " + dates[0] + " and " + dates[-1])
        series = database[i][initial_index: (final_index + 1)]

        print("You opted for the analysis of the " + input_type)
        if input_type == "Returns":
            X = np.array(series[tau:]) - \
                np.array(series[0: (len(series) - tau)])
            lab = "P(t+" + str(tau) + ") - P(t)"
        elif input_type == "Relative returns":
            X = np.array(series[tau:]) / \
                np.array(series[0: (len(series) - tau)]) - 1.0
            lab = "P(t+" + str(tau) + ")/P(t) - 1.0"
        else:
            X = np.log(
                np.array(series[tau:]) /
                np.array(series[0: (len(series) - tau)])
            )
            lab = r"$\log$" + "(P(t+" + str(tau) + ")/P(t))"

        #  if standardize == "Yes":
        #      if standardize_target == "Full Series":
        #          print("I am standardizing your time series")
        #          S = X
        #          m = np.mean(S)
        #          v = np.std(S)
        #          X = (S - m) / v

        #  if abs_value == "Yes":
        #      if abs_target == "Full Series":
        #          print("I am taking the absolute value of your time series")
        #          X = np.abs(X)
        #          lab = "|" + lab + "|"

        if tail_selected == "Right" or tail_selected == "Both":
            tail_plus = X
            if xmin_rule == "Clauset":
                fit_1 = PowerLawFit(tail_plus, data_nature, xmin_rule)
            #  elif xmin_rule == "Manual":
            #      fit_1 = PowerLawFit(tail_plus, data_nature,
            #                          xmin_rule, xmin_value)
            #  else:
            #      fit_1 = PowerLawFit(
            #          tail_plus, data_nature, xmin_rule, "None", xmin_sign
            #      )

            #  if standardize == "Yes":
            #      if standardize_target == "Tail":
            #          print("I am standardizing your tail")
            #          S = np.array(filter(lambda x: x >= fit_1.power_law.xmin,
            #                              tail_plus))
            #          m = np.mean(S)
            #          v = np.std(S)
            #          X = (S - m) / v
            #          if abs_value == "Yes":
            #              if abs_target == "Tail":
            #                  print("I am taking the absolute "
            #                        " value of your tail")
            #                  X = np.abs(X)
            #                  lab = "|" + lab + "|"
            #          fit_1 =PowerLawFit(X, data_nature, xmin_rule, np.min(X))

        if tail_selected == "Left" or tail_selected == "Both":
            tail_neg = (np.dot(-1.0, X)).tolist()
            if xmin_rule == "Clauset":
                fit_2 = PowerLawFit(tail_neg, data_nature, xmin_rule)
            #  elif xmin_rule == "Manual":
            #      fit_2 = PowerLawFit(tail_neg, data_nature,
            #                          xmin_rule, xmin_value)
            #  else:
            #      fit_2 = PowerLawFit(tail_neg, data_nature,
            #                          xmin_rule, "None", xmin_sign)

            #  if standardize == "Yes":
            #      if standardize_target == "Tail":
            #          print("I am standardizing your tail")
            #          S = np.array(filter(lambda x: x >= fit_2.power_law.xmin,
            #                              tail_neg))
            #          m = np.mean(S)
            #          v = np.std(S)
            #          X = (S - m) / v
            #          if abs_value == "Yes":
            #              if abs_target == "Tail":
            #                  print("I am taking the absolute "
            #                        "value of your tail")
            #                  X = np.abs(X)
            #                  lab = "|" + lab + "|"
            #          fit_2 =PowerLawFit(X, data_nature, xmin_rule, np.min(X))

        if tail_selected == "Right" or tail_selected == "Both":
            alpha1 = fit_1.power_law.alpha
            xmin1 = fit_1.power_law.xmin
            s_err1 = fit_1.power_law.sigma
            p1 = plpva.plpva(tail_plus.tolist(), float(xmin1),
                             "reps", c_iter, "silent")
            #  positive_alpha_KS.append(p1[0])
            results["pos_α_ks"].append(p1[0])

        if tail_selected == "Left" or tail_selected == "Both":
            alpha2 = fit_2.power_law.alpha
            xmin2 = fit_2.power_law.xmin
            s_err2 = fit_2.power_law.sigma
            p2 = plpva.plpva(np.array(tail_neg).tolist(),
                             float(xmin2), "reps", c_iter, "silent")
            #  negative_alpha_KS.append(p2[0])
            results["neg_α_ks"].append(p2[0])

        if tail_selected == "Right" or tail_selected == "Both":

            plt.figure("Right tail scaling for " + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            fig4 = fit_1.plot_ccdf(color="b", linewidth=2,
                                   label="Empirical CCDF")
            fit_1.power_law.plot_ccdf(
                color="b", linestyle="-", label="Fitted CCDF", ax=fig4
            )
            fit_1.plot_pdf(color="r", linewidth=2,
                           label="Empirical PDF", ax=fig4)
            fit_1.power_law.plot_pdf(
                color="r", linestyle="-", label="Fitted PDF", ax=fig4
            )
            fig4.set_title(
                "Log-log plot of the scaling properties of the right-tail for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            fig4.grid()
            fig4.legend()
            col_labels = [r"$\hat{\alpha}$", "Standard err.",
                          r"$x_{min}$", "size"]
            table_vals = []
            table_vals.append(
                [
                    np.round(alpha1, 4),
                    np.round(s_err1, 4),
                    np.round(xmin1, 4),
                    len(filter(lambda x: x > xmin1, tail_plus)),
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
            plt.show()

            plt.figure("Right tail comparison for " + labels[i - 1])
            fig4 = fit_1.plot_ccdf(color="b", linewidth=2,
                                   label="Empirical CCDF")
            fit_1.power_law.plot_ccdf(
                color="r", linestyle="-", label="Fitted PL", ax=fig4
            )
            fit_1.truncated_power_law.plot_ccdf(
                color="g", linestyle="-", label="Fitted TPL", ax=fig4
            )
            fit_1.exponential.plot_ccdf(
                color="c", linestyle="-", label="Fitted Exp.", ax=fig4
            )
            fit_1.lognormal.plot_ccdf(
                color="m", linestyle="-", label="Fitted LogN.", ax=fig4
            )
            fig4.set_title(
                "Comparison of the distributions fitted on the right-tail for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            fig4.grid()
            fig4.legend()
            plt.show()

            distribution_list = ["truncated_power_law",
                                 "exponential", "lognormal"]
            for pdf in distribution_list:
                R, p = fit_1.distribution_compare(
                    "power_law", pdf, normalized_ratio=True)
                #  loglikelihood_ratio_right.append(R)
                results["loglr_right"].append(R)
                #  loglikelihood_pvalue_right.append(p)
                results["loglpv_right"].append(p)

            z.figure("Log Likelihood ratio for the right tail for " +
                     labels[i - 1])
            #  z.bar(
            #      np.arange(0, len(loglikelihood_ratio_right), 1),
            #      loglikelihood_ratio_right, 1,)
            z.bar(
                np.arange(0, len(results["loglr_right"]), 1),
                results["loglr_right"], 1,)
            z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
                     distribution_list)
            z.ylabel("R")
            z.title(
                "Log-likelihood ratio for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            z.grid()
            # z.show()

            z.figure("Log Likelihood ratio p-values for the right tail for " +
                     labels[i - 1])
            #  z.bar(
            #      np.arange(0, len(loglikelihood_pvalue_right), 1),
            #      loglikelihood_pvalue_right, 1,)
            z.bar(
                np.arange(0, len(results["loglpv_right"]), 1),
                results["loglpv_right"], 1,)
            z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
                     distribution_list)
            z.ylabel("R")
            z.title(
                "Log-likelihood ratio p values for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            z.grid()
            # z.show()

        if tail_selected == "Left" or tail_selected == "Both":

            plt.figure("Left tail scaling for " + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            fig4 = fit_2.plot_ccdf(color="b", linewidth=2,
                                   label="Empirical CCDF")
            fit_2.power_law.plot_ccdf(
                color="b", linestyle="-", label="Fitted CCDF", ax=fig4
            )
            fit_2.plot_pdf(color="r", linewidth=2,
                           label="Empirical PDF", ax=fig4)
            fit_2.power_law.plot_pdf(
                color="r", linestyle="-", label="Fitted PDF", ax=fig4
            )
            fig4.set_title(
                "Log-log plot of the scaling properties of the left-tail for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            fig4.grid()
            fig4.legend()
            col_labels = [r"$\hat{\alpha}$", "Standard err.",
                          r"$x_{min}$", "size"]
            table_vals = []
            table_vals.append(
                [
                    np.round(alpha2, 4),
                    np.round(s_err2, 4),
                    np.round(xmin2, 4),
                    len(filter(lambda x: x > xmin2, tail_neg)),
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
            plt.show()

            plt.figure("Left tail comparison for " + labels[i - 1])
            fig4 = fit_2.plot_ccdf(color="b", linewidth=2,
                                   label="Empirical CCDF")
            fit_2.power_law.plot_ccdf(
                color="r", linestyle="-", label="Fitted PL", ax=fig4
            )
            fit_2.truncated_power_law.plot_ccdf(
                color="g", linestyle="-", label="Fitted TPL", ax=fig4
            )
            fit_2.exponential.plot_ccdf(
                color="c", linestyle="-", label="Fitted Exp.", ax=fig4
            )
            fit_2.lognormal.plot_ccdf(
                color="m", linestyle="-", label="Fitted LogN.", ax=fig4
            )
            fig4.set_title(
                "Comparison of the distributions fitted on the left-tail for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            fig4.grid()
            fig4.legend()
            plt.show()

            distribution_list = ["truncated_power_law",
                                 "exponential", "lognormal"]
            for pdf in distribution_list:
                R, p = fit_2.distribution_compare(
                    "power_law", pdf, normalized_ratio=True
                )
                #  loglikelihood_ratio_left.append(R)
                results["loglr_left"].append(R)
                #  loglikelihood_pvalue_left.append(p)
                results["loglpv_left"].append(p)

            z.figure("Log Likelihood ratio for the left tail for " +
                     labels[i - 1])
            #  z.bar(
            #      np.arange(0, len(loglikelihood_ratio_left), 1),
            #      loglikelihood_ratio_left, 1,)
            z.bar(
                np.arange(0, len(results["loglr_left"]), 1),
                results["loglr_left"], 1,)
            z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
                     distribution_list)
            z.ylabel("R")
            z.title(
                "Log-likelihood ratio for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            z.grid()
            # z.show()

            z.figure(
                "Log Likelihood ratio p-values for the left tail for " +
                labels[i - 1])
            #  z.bar(
            #      np.arange(0, len(loglikelihood_pvalue_left), 1),
            #      loglikelihood_pvalue_left, 1,)
            z.bar(
                np.arange(0, len(results["loglpv_left"]), 1),
                results["loglpv_left"], 1,)
            z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
                     distribution_list)
            z.ylabel("R")
            z.title(
                "Log-likelihood ratio p values for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input series: "
                + lab
            )
            z.grid()
            # z.show()

        if tail_selected == "Right" or tail_selected == "Both":

            #  positive_alpha_vec.append(alpha1)
            results["pos_α_vec"].append(alpha1)
            #  positive_upper_bound.append(
            #      alpha1 + (st.norm.ppf(1-multiplier * significance))*s_err1)
            results["pos_up_bound"].append(
                alpha1 + (st.norm.ppf(1 - multiplier * significance)) * s_err1)
            #  positive_lower_bound.append(
            #      alpha1 - (st.norm.ppf(1-multiplier * significance))*s_err1)
            results["pos_low_bound"].append(
                alpha1 - (st.norm.ppf(1 - multiplier * significance)) * s_err1)
            #  positive_abs_length.append(len(filter(lambda x: x >= xmin1,
            #                                        tail_plus)))
            results["pos_abs_len"].append(len(filter(lambda x: x >= xmin1,
                                                     tail_plus)))
            #  positive_rel_length.append(
            #      len(filter(lambda x: x >= xmin1, tail_plus)) /
            #      float(len(tail_plus)))
            results["pos_rel_len"].append(
                len(filter(lambda x: x >= xmin1, tail_plus)) /
                float(len(tail_plus)))

        if tail_selected == "Left" or tail_selected == "Both":
            #  negative_alpha_vec.append(alpha2)
            results["neg_α_vec"].append(alpha2)
            #  negative_upper_bound.append(
            #      alpha2 + (st.norm.ppf(1-multiplier * significance))*s_err2)
            results["neg_up_bound"].append(
                alpha2 + (st.norm.ppf(1 - multiplier * significance)) * s_err2)
            #  negative_lower_bound.append(
            #      alpha2 - (st.norm.ppf(1-multiplier * significance))*s_err2)
            results["neg_low_bound"].append(
                alpha2 - (st.norm.ppf(1 - multiplier * significance)) * s_err2)
            #  negative_abs_length.append(len(filter(lambda x: x >= xmin2,
            #                                        tail_neg)))
            results["neg_abs_len"].append(len(filter(lambda x: x >= xmin2,
                                                     tail_neg)))
            #  negative_rel_length.append(
            #      len(filter(lambda x: x >= xmin2, tail_neg)) /
            #      float(len(tail_neg)))
            results["neg_rel_len"].append(
                len(filter(lambda x: x >= xmin2, tail_neg)) /
                float(len(tail_neg)))

        if tail_selected == "Both":
            row = [
                alpha1,
                alpha2,
                xmin1,
                xmin2,
                s_err1,
                s_err2,
                len(filter(lambda x: x >= xmin1, tail_plus)),
                len(filter(lambda x: x >= xmin2, tail_neg)),
                p1[0],
                p2[0],
            ]
        if tail_selected == "Right":
            row = [
                alpha1,
                0,
                xmin1,
                0,
                s_err1,
                0,
                len(filter(lambda x: x >= xmin1, tail_plus)),
                0,
                p1[0],
                0,
            ]
        if tail_selected == "Left":
            row = [
                0,
                alpha2,
                0,
                xmin2,
                0,
                s_err2,
                0,
                len(filter(lambda x: x >= xmin2, tail_neg)),
                0,
                p2[0],
            ]

        tail_statistics.append(row)

    # Preparing the figure

    z.figure("Static alpha")
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    if tail_selected == "Right" or tail_selected == "Both":
        z.plot(
            range(1, len(labels) + 1, 1),
            #  positive_alpha_vec,
            results["pos_α_vec"],
            marker="^",
            markersize=10.0,
            linewidth=0.0,
            color="green",
            label="Right tail",
        )
    if tail_selected == "Left" or tail_selected == "Both":
        z.plot(
            range(1, len(labels) + 1, 1),
            #  negative_alpha_vec,
            results["neg_α_vec"],
            marker="^",
            markersize=10.0,
            linewidth=0.0,
            color="red",
            label="Left tail",
        )
    z.xticks(range(1, len(labels) + 1, 1), labels)
    z.xlim(xmin=0.5, xmax=len(labels) + 0.5)
    z.ylabel(r"$\alpha$")
    z.title(
        "Estimation of the "
        + r"$\alpha$"
        + "-right tail exponents using KS-Method"
        + "\n"
        + "Time Period: "
        + dates[0]
        + " - "
        + dates[-1]
        + ". Input series: "
        + lab
    )
    z.legend(
        bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=2,
        mode="expand", borderaxespad=0
    )
    z.grid()
    # z.show()

    if tail_selected == "Right" or tail_selected == "Both":

        # Confidence interval for the right tail
        z.figure("Confidence interval for the right tail")
        z.gca().set_position((0.1, 0.20, 0.83, 0.70))
        z.plot(
            range(1, len(labels) + 1, 1),
            #  positive_alpha_vec,
            results["pos_α_vec"],
            marker="o",
            markersize=7.0,
            linewidth=0.0,
            color="green",
            label="Right tail",
        )
        z.plot(
            range(1, len(labels) + 1, 1),
            #  positive_upper_bound,
            results["pos_up_bound"],
            marker="o",
            markersize=7.0,
            linewidth=0.0,
            color="purple",
            label="Upper bound",
        )
        z.plot(
            range(1, len(labels) + 1, 1),
            #  positive_lower_bound,
            results["pos_low_bound"],
            marker="o",
            markersize=7.0,
            linewidth=0.0,
            color="blue",
            label="Lower bound",
        )
        z.plot(
            range(0, len(labels) + 2, 1), np.repeat(3, len(labels) + 2),
            color="red"
        )
        z.plot(
            range(0, len(labels) + 2, 1), np.repeat(2, len(labels) + 2),
            color="red"
        )
        z.xticks(range(1, len(labels) + 1, 1), labels)
        z.xlim(xmin=0.5, xmax=len(labels) + 0.5)
        z.ylabel(r"$\alpha$")
        z.title(
            "Confidence intervals for the "
            + r"$\alpha$"
            + "-right tail exponents "
            + "(c = "
            + str(1 - significance)
            + ")"
            + "\n"
            + "Time Period: "
            + dates[0]
            + " - "
            + dates[-1]
            + ". Input series: "
            + lab
        )
        z.legend(
            bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
            ncol=3,
            mode="expand",
            borderaxespad=0,
        )
        z.grid()
        # z.show()

    if tail_selected == "Left" or tail_selected == "Both":

        # Confidence interval for the left tail
        z.figure("Confidence interval for the left tail")
        z.gca().set_position((0.1, 0.20, 0.83, 0.70))
        z.plot(
            range(1, len(labels) + 1, 1),
            #  negative_alpha_vec,
            results["neg_α_vec"],
            marker="o",
            markersize=7.0,
            linewidth=0.0,
            color="green",
            label="Left tail",
        )
        z.plot(
            range(1, len(labels) + 1, 1),
            #  negative_upper_bound,
            results["neg_up_bound"],
            marker="o",
            markersize=7.0,
            linewidth=0.0,
            color="purple",
            label="Upper bound",
        )
        z.plot(
            range(1, len(labels) + 1, 1),
            #  negative_lower_bound,
            results["neg_low_bound"],
            marker="o",
            markersize=7.0,
            linewidth=0.0,
            color="blue",
            label="Lower bound",
        )
        z.plot(
            range(0, len(labels) + 2, 1), np.repeat(3, len(labels) + 2),
            color="red"
        )
        z.plot(
            range(0, len(labels) + 2, 1), np.repeat(2, len(labels) + 2),
            color="red"
        )
        z.xticks(range(1, len(labels) + 1, 1), labels)
        z.xlim(xmin=0.5, xmax=len(labels) + 0.5)
        z.ylabel(r"$\alpha$")
        z.title(
            "Confidence intervals for the "
            + r"$\alpha$"
            + "-left tail exponents "
            + "(c = "
            + str(1 - significance)
            + ")"
            + "\n"
            + "Time Period: "
            + dates[0]
            + " - "
            + dates[-1]
            + ". Input series: "
            + lab
        )
        z.legend(
            bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
            ncol=3,
            mode="expand",
            borderaxespad=0,
        )
        z.grid()
        # z.show()

    # Absolute length of the tail bar chart

    z.figure("Absolute tail lengths")
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    amplitude = 0.5
    if tail_selected == "Right" or tail_selected == "Both":
        z.bar(
            np.arange(0, 2 * len(labels), 2),
            #  positive_abs_length,
            results["pos_abs_len"],
            amplitude,
            color="green",
            label="Right tail",
        )
    if tail_selected == "Left" or tail_selected == "Both":
        z.bar(
            np.arange(amplitude, 2 * len(labels) + amplitude, 2),
            #  negative_abs_length,
            results["neg_abs_len"],
            amplitude,
            color="red",
            label="Left tail",
        )
    z.xticks(np.arange(amplitude, 2 * len(labels) + amplitude, 2), labels)
    z.ylabel("Tail length")
    z.title(
        "Bar chart representation of the length of the tails"
        + "\n"
        + "Time Period: "
        + dates[0]
        + " - "
        + dates[-1]
        + ". Input series: "
        + lab
    )
    z.legend(
        bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3, mode="expand",
        borderaxespad=0
    )
    z.grid()
    # z.show()

    # Absolute length of the tail bar chart

    z.figure("Relative tail lengths")
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    amplitude = 0.5
    if tail_selected == "Right" or tail_selected == "Both":
        z.bar(
            np.arange(0, 2 * len(labels), 2),
            #  positive_rel_length,
            results["pos_rel_len"],
            amplitude,
            color="green",
            label="Right tail",
        )
    if tail_selected == "Left" or tail_selected == "Both":
        z.bar(
            np.arange(amplitude, 2 * len(labels) + amplitude, 2),
            #  negative_rel_length,
            results["neg_rel_len"],
            amplitude,
            color="red",
            label="Left tail",
        )
    z.xticks(np.arange(amplitude, 2 * len(labels) + amplitude, 2), labels)
    z.ylabel("Tail relative length")
    z.title(
        "Bar chart representation of the relative length of the tails"
        + "\n"
        + "Time Period: "
        + dates[0]
        + " - "
        + dates[-1]
        + ". Input series: "
        + lab
    )
    z.legend(
        bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3,
        mode="expand", borderaxespad=0
    )
    z.grid()
    # z.show()

    # KS test outcome

    z.figure("KS test p value for the tails")
    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    amplitude = 0.5
    if tail_selected == "Right" or tail_selected == "Both":
        z.bar(
            np.arange(0, 2 * len(labels), 2),
            #  positive_alpha_KS,
            results["pos_α_ks"],
            amplitude,
            color="green",
            label="Right tail",
        )
    if tail_selected == "Left" or tail_selected == "Both":
        z.bar(
            np.arange(amplitude, 2 * len(labels) + amplitude, 2),
            #  negative_alpha_KS,
            results["neg_α_ks"],
            amplitude,
            color="red",
            label="Left tail",
        )
    z.xticks(np.arange(amplitude, 2 * len(labels) + amplitude, 2), labels)
    z.ylabel("p-value")
    z.title(
        "KS-statistics: p-value obtained from Clauset algorithm"
        + "\n"
        + "Time Period: "
        + dates[0]
        + " - "
        + dates[-1]
        + ". Input series: "
        + lab
    )
    z.legend(
        bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3,
        mode="expand", borderaxespad=0
    )
    z.grid()
    # z.show()

    # Print the figures
    matrix_form = np.array(tail_statistics)
    matrix_form_transpose = np.transpose(matrix_form)
    filename = "TailStatistics_Overall.csv"
    df = pd.DataFrame(
        {
            "Input": labels,
            "Positive Tail Exponent": matrix_form_transpose[0],
            "Negative Tail Exponent": matrix_form_transpose[1],
            "Positive Tail xmin": matrix_form_transpose[2],
            "Negative Tail xmin": matrix_form_transpose[3],
            "Positive Tail S.Err": matrix_form_transpose[4],
            "Negative Tail S.Err": matrix_form_transpose[5],
            "Positive Tail Size": matrix_form_transpose[6],
            "Negative Tail Size": matrix_form_transpose[7],
            "Positive Tail KS p-value": matrix_form_transpose[8],
            "Negative Tail KS p-value": matrix_form_transpose[9],
        }
    )

    df = df[
        [
            "Input",
            "Positive Tail Exponent",
            "Negative Tail Exponent",
            "Positive Tail xmin",
            "Negative Tail xmin",
            "Positive Tail S.Err",
            "Negative Tail S.Err",
            "Positive Tail Size",
            "Negative Tail Size",
            "Positive Tail KS p-value",
            "Negative Tail KS p-value",
        ]
    ]
    df.to_csv(filename, index=False)

else:

    #  question      = "Do you want to save the sequential scaling plot?"
    #  choices      = ['Yes', 'No']
    #  plot_storing = eg.choicebox(question, 'Plot', choices)
    plot_storing = "No"

    #  if plot_storing == "Yes":
    #      question = "What is the target directory for the pictures?"
    #      motherpath = eg.enterbox(
    #          question,
    #          title="path",
    #          default = ("C:\Users\\alber\Dropbox\Research"
    #                     "\IP\Econophysics\Final Code Hurst Exponent\\"),
    #      )

    initial_index = database[0].index(initial_date)
    final_index = database[0].index(final_date)
    dates = database[0][initial_index: (final_index + 1)]
    labelstep = (
        22
        if len(dates) <= 252
        else 66
        if (len(dates) > 252 and len(dates) <= 756)
        else 121
    )
    N = len(database)

    temp = []

    if an_freq > 1:
        spec_dates = []
        for ddd in range(0, len(dates), an_freq):
            spec_dates.append(dates[ddd])
        spec_labelstep = 22
    else:
        spec_dates = dates
        spec_labelstep = labelstep

    # int(np.maximum(np.floor(22/float(an_freq)),1.0))

    # TODO: add lists below to results_lists_init function
    positive_alpha_mat = []
    negative_alpha_mat = []

    for i in range(1, N, 1):

        #  if plot_storing == "Yes":
        #      directory = motherpath + "PowerLawAnimation\\" + labels[i - 1]
        #      try:
        #          os.makedirs(directory)
        #      except OSError:
        #          if not os.path.isdir(directory):
        #              raise
        #      os.chdir(directory)

        #  positive_alpha_vec = []
        #  negative_alpha_vec = []
        #  positive_alpha_KS = []
        #  negative_alpha_KS = []
        #  positive_upper_bound = []
        #  positive_lower_bound = []
        #  negative_upper_bound = []
        #  negative_lower_bound = []
        #  positive_abs_length = []
        #  positive_rel_length = []
        #  negative_abs_length = []
        #  negative_rel_length = []
        #  loglikelihood_ratio_right = []
        #  loglikelihood_pvalue_right = []
        #  loglikelihood_ratio_left = []
        #  loglikelihood_pvalue_left = []

        # TODO: add list below to results_lists_init function
        tail_statistics = []

        for l in range(initial_index, final_index + 1, an_freq):

            if approach == "Rolling":
                series = database[i][(l + 1 - lookback): (l + 1)]
                begin_date = database[0][(l + 1 - lookback)]
                end_date = database[0][l]
            else:
                series = database[i][(initial_index + 1 - lookback): (l + 1)]
                begin_date = database[0][(initial_index + 1 - lookback)]
                end_date = database[0][l]

            #  if plot_storing == "Yes":
            #      subdirectory = (
            #          directory
            #          + "\\"
            #          + begin_date[6:8]
            #          + "-"
            #          + begin_date[3:5]
            #          + "-"
            #          + begin_date[0:2]
            #          + "_"
            #          + end_date[6:8]
            #          + "-"
            #          + end_date[3:5]
            #          + "-"
            #          + end_date[0:2]
            #          + "\\"
            #      )
            #      try:
            #          os.makedirs(subdirectory)
            #      except OSError:
            #          if not os.path.isdir(subdirectory):
            #              raise
            #      os.chdir(subdirectory)

            print("I am analyzing the time series for " +
                  labels[i - 1] + " between " + begin_date +
                  " and " + end_date)

            print("You opted for the analysis of the " + input_type)

            if input_type == "Returns":
                X = np.array(series[tau:]) - np.array(
                    series[0: (len(series) - tau)])
                lab = "P(t+" + str(tau) + ") - P(t)"
            elif input_type == "Relative returns":
                X = (
                    np.array(series[tau:]) /
                    np.array(series[0: (len(series) - tau)])
                    - 1.0
                )
                lab = "P(t+" + str(tau) + ")/P(t) - 1.0"
            else:
                X = np.log(
                    np.array(series[tau:]) /
                    np.array(series[0: (len(series) - tau)])
                )
                lab = r"$\log$" + "(P(t+" + str(tau) + ")/P(t))"

            if standardize == "Yes":
                print("I am standardizing your time series")
                S = X
                m = np.mean(S)
                v = np.std(S)
                X = (S - m) / v

            if abs_value == "Yes":
                print("I am taking the absolute value of your time series")
                X = np.abs(X)
                lab = "|" + lab + "|"

            if tail_selected == "Right" or tail_selected == "Both":
                tail_plus = X
                if xmin_rule == "Clauset":
                    fit_1 = PowerLawFit(tail_plus, data_nature, xmin_rule)
                #  elif xmin_rule == "Manual":
                #      fit_1 = PowerLawFit(tail_plus, data_nature,
                #                          xmin_rule, xmin_value)
                #  else:
                #      fit_1 = PowerLawFit(
                #          tail_plus, data_nature, xmin_rule, "None", xmin_sign
                #      )

                alpha1 = fit_1.power_law.alpha
                xmin1 = fit_1.power_law.xmin
                s_err1 = fit_1.power_law.sigma

            if tail_selected == "Left" or tail_selected == "Both":
                tail_neg = (np.dot(-1.0, X)).tolist()
                if xmin_rule == "Clauset":
                    fit_2 = PowerLawFit(tail_neg, data_nature, xmin_rule)
                #  elif xmin_rule == "Manual":
                #      fit_2 = PowerLawFit(tail_neg, data_nature,
                #                          xmin_rule, xmin_value)
                #  else:
                #      fit_2 = PowerLawFit(
                #          tail_neg, data_nature, xmin_rule, "None", xmin_sign
                #      )

                alpha2 = fit_2.power_law.alpha
                xmin2 = fit_2.power_law.xmin
                s_err2 = fit_2.power_law.sigma

            if plot_storing == "Yes":

                if tail_selected == "Right" or tail_selected == "Both":

                    plt.figure(
                        "Right tail scaling for "
                        + labels[i - 1]
                        + begin_date
                        + "_"
                        + end_date
                    )
                    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
                    fig4 = fit_1.plot_ccdf(
                        color="b", linewidth=2, label="Empirical CCDF"
                    )
                    fit_1.power_law.plot_ccdf(
                        color="b", linestyle="-", label="Fitted CCDF", ax=fig4
                    )
                    fit_1.plot_pdf(
                        color="r", linewidth=2, label="Empirical PDF", ax=fig4
                    )
                    fit_1.power_law.plot_pdf(
                        color="r", linestyle="-", label="Fitted PDF", ax=fig4
                    )
                    fig4.set_title(
                        "Log-log plot of the scaling properties "
                        "of the right-tail for "
                        + labels[i - 1]
                        + "\n"
                        + "Time Period: "
                        + begin_date
                        + " - "
                        + end_date
                        + ". Input series: "
                        + lab
                    )
                    fig4.grid()
                    fig4.legend()
                    col_labels = [
                        r"$\hat{\alpha}$",
                        "Standard err.",
                        r"$x_{min}$",
                        "size",
                    ]
                    table_vals = []
                    table_vals.append(
                        [
                            np.round(alpha1, 4),
                            np.round(s_err1, 4),
                            np.round(xmin1, 4),
                            len(filter(lambda x: x >= xmin1, tail_plus)),
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
                    plt.savefig(
                        "Right-tail scaling_"
                        + begin_date
                        + "_"
                        + end_date
                        + "_"
                        + labels[i - 1]
                        + ".jpg"
                    )
                    plt.close()

                    plt.figure("Right tail comparison for " + labels[i - 1])
                    fig4 = fit_1.plot_ccdf(
                        color="b", linewidth=2, label="Empirical CCDF"
                    )
                    fit_1.power_law.plot_ccdf(
                        color="r", linestyle="-", label="Fitted PL", ax=fig4
                    )
                    fit_1.truncated_power_law.plot_ccdf(
                        color="g", linestyle="-", label="Fitted TPL", ax=fig4
                    )
                    fit_1.exponential.plot_ccdf(
                        color="c", linestyle="-", label="Fitted Exp.", ax=fig4
                    )
                    fit_1.lognormal.plot_ccdf(
                        color="m", linestyle="-", label="Fitted LogN.", ax=fig4
                    )
                    fig4.set_title(
                        "Comparison of the distributions "
                        "fitted on the right-tail for "
                        + labels[i - 1]
                        + "\n"
                        + "Time Period: "
                        + dates[0]
                        + " - "
                        + dates[-1]
                        + ". Input series: "
                        + lab
                    )
                    fig4.grid()
                    fig4.legend()
                    plt.savefig(
                        "Right-tail fitting comparison_"
                        + begin_date
                        + "_"
                        + end_date
                        + "_"
                        + labels[i - 1]
                        + ".jpg"
                    )
                    plt.close()

                if tail_selected == "Left" or tail_selected == "Both":

                    plt.figure(
                        "Left tail scaling for "
                        + labels[i - 1]
                        + begin_date
                        + "_"
                        + end_date
                    )
                    z.gca().set_position((0.1, 0.20, 0.83, 0.70))
                    fig4 = fit_2.plot_ccdf(
                        color="b", linewidth=2, label="Empirical CCDF"
                    )
                    fit_2.power_law.plot_ccdf(
                        color="b", linestyle="-", label="Fitted CCDF", ax=fig4
                    )
                    fit_2.plot_pdf(
                        color="r", linewidth=2, label="Empirical PDF", ax=fig4
                    )
                    fit_2.power_law.plot_pdf(
                        color="r", linestyle="-", label="Fitted PDF", ax=fig4
                    )
                    fig4.set_title(
                        "Log-log plot of the scaling properties "
                        "of the left-tail for "
                        + labels[i - 1]
                        + "\n"
                        + "Time Period: "
                        + begin_date
                        + " - "
                        + end_date
                        + ". Input series: "
                        + lab
                    )
                    fig4.grid()
                    fig4.legend()
                    col_labels = [
                        r"$\hat{\alpha}$",
                        "Standard err.",
                        r"$x_{min}$",
                        "size",
                    ]
                    table_vals = []
                    table_vals.append(
                        [
                            np.round(alpha2, 4),
                            np.round(s_err2, 4),
                            np.round(xmin2, 4),
                            len(filter(lambda x: x >= xmin2, tail_neg)),
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
                    plt.savefig(
                        "Left-tail scaling_"
                        + begin_date
                        + "_"
                        + end_date
                        + "_"
                        + labels[i - 1]
                        + ".jpg"
                    )
                    plt.close()

                    plt.figure("Left tail comparison for " + labels[i - 1])
                    fig4 = fit_2.plot_ccdf(
                        color="b", linewidth=2, label="Empirical CCDF"
                    )
                    fit_2.power_law.plot_ccdf(
                        color="r", linestyle="-", label="Fitted PL", ax=fig4
                    )
                    fit_2.truncated_power_law.plot_ccdf(
                        color="g", linestyle="-", label="Fitted TPL", ax=fig4
                    )
                    fit_2.exponential.plot_ccdf(
                        color="c", linestyle="-", label="Fitted Exp.", ax=fig4
                    )
                    fit_2.lognormal.plot_ccdf(
                        color="m", linestyle="-", label="Fitted LogN.", ax=fig4
                    )
                    fig4.set_title(
                        "Comparison of the distributions fitted "
                        "on the left-tail for "
                        + labels[i - 1]
                        + "\n"
                        + "Time Period: "
                        + dates[0]
                        + " - "
                        + dates[-1]
                        + ". Input series: "
                        + lab
                    )
                    fig4.grid()
                    fig4.legend()
                    plt.savefig(
                        "Left-tail fitting comparison_"
                        + begin_date
                        + "_"
                        + end_date
                        + "_"
                        + labels[i - 1]
                        + ".jpg"
                    )
                    plt.close()

            if tail_selected == "Right" or tail_selected == "Both":

                results["pos_α_vec"].append(alpha1)
                results["pos_up_bound"].append(
                    alpha1 + (st.norm.ppf(1 - multiplier * significance))
                    * s_err1)
                results["pos_low_bound"].append(
                    alpha1 - (st.norm.ppf(1 - multiplier * significance))
                    * s_err1
                )
                #  positive_abs_length.append(len(filter(lambda x: x >= xmin1,
                #                                        tail_plus)))
                results["pos_abs_len"].append(len(
                    tail_plus[tail_plus >= xmin1]))
                results["pos_rel_len"].append(
                    len(tail_plus[tail_plus >= xmin1]) /
                    float(len(tail_plus)))
                p1 = plpva.plpva(
                    tail_plus.tolist(), float(xmin1), "reps", c_iter, "silent"
                )
                results["pos_α_ks"].append(p1[0])

                distribution_list = ["truncated_power_law",
                                     "exponential", "lognormal"]
                daily_r_ratio = []
                daily_r_p = []
                for pdf in distribution_list:
                    R, p = fit_1.distribution_compare(
                        "power_law", pdf, normalized_ratio=True
                    )
                    daily_r_ratio.append(R)
                    daily_r_p.append(p)

                results["loglr_right"].append(daily_r_ratio)
                results["loglpv_right"].append(daily_r_p)

            if tail_selected == "Left" or tail_selected == "Both":

                results["neg_α_vec"].append(alpha2)
                results["neg_up_bound"].append(
                    alpha2 + (st.norm.ppf(1 - multiplier * significance))
                    * s_err2
                )
                results["neg_low_bound"].append(
                    alpha2 - (st.norm.ppf(1 - multiplier * significance))
                    * s_err2
                )
                #  negative_abs_length.append(len(filter(lambda x: x >= xmin2,
                #                                        tail_neg)))
                # NOTE: tail_plus was already converted
                tail_neg = np.array(tail_neg)
                results["neg_abs_len"].append(len(tail_neg[tail_neg >= xmin2]))
                results["neg_rel_len"].append(
                    len(tail_neg[tail_neg >= xmin2]) /
                    float(len(tail_neg)))
                p2 = plpva.plpva(
                    np.array(tail_neg).tolist(), float(xmin2), "reps",
                    c_iter, "silent"
                )
                results["neg_α_ks"].append(p2[0])

                distribution_list = ["truncated_power_law",
                                     "exponential", "lognormal"]
                daily_l_ratio = []
                daily_l_p = []
                for pdf in distribution_list:
                    R, p = fit_2.distribution_compare(
                        "power_law", pdf, normalized_ratio=True
                    )
                    daily_l_ratio.append(R)
                    daily_l_p.append(p)

                results["loglr_left"].append(daily_l_ratio)
                results["loglpv_left"].append(daily_l_p)

            if tail_selected == "Both":
                row = [
                    alpha1,
                    alpha2,
                    xmin1,
                    xmin2,
                    s_err1,
                    s_err2,
                    #  len(filter(lambda x: x >= xmin1, tail_plus)),
                    #  len(filter(lambda x: x >= xmin2, tail_neg)),
                    len(tail_plus[tail_plus >= xmin1]),
                    len(tail_neg[tail_neg >= xmin2]),
                    p1[0],
                    p2[0],
                    daily_r_ratio[0],
                    daily_r_ratio[1],
                    daily_r_ratio[2],
                    daily_r_p[0],
                    daily_r_p[1],
                    daily_r_p[2],
                    daily_l_ratio[0],
                    daily_l_ratio[1],
                    daily_l_ratio[2],
                    daily_l_p[0],
                    daily_l_p[1],
                    daily_l_p[2],
                ]
            if tail_selected == "Right":
                row = [
                    alpha1,
                    0,
                    xmin1,
                    0,
                    s_err1,
                    0,
                    len(filter(lambda x: x >= xmin1, tail_plus)),
                    0,
                    p1[0],
                    0,
                    daily_r_ratio[0],
                    daily_r_ratio[1],
                    daily_r_ratio[2],
                    daily_r_p[0],
                    daily_r_p[1],
                    daily_r_p[2],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            if tail_selected == "Left":
                row = [
                    0,
                    alpha2,
                    0,
                    xmin2,
                    0,
                    s_err2,
                    0,
                    len(filter(lambda x: x >= xmin2, tail_neg)),
                    0,
                    p2[0],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    daily_l_ratio[0],
                    daily_l_ratio[1],
                    daily_l_ratio[2],
                    daily_l_p[0],
                    daily_l_p[1],
                    daily_l_p[2],
                ]

            tail_statistics.append(row)

        if tail_selected == "Right" or tail_selected == "Both":
            positive_alpha_mat.append(results["pos_α_vec"])
        if tail_selected == "Left" or tail_selected == "Both":
            negative_alpha_mat.append(results["neg_α_vec"])

        # Plot the alpha exponent in time (right/left/both tail)

        z.figure("Alpha Fitting for " + labels[i - 1])
        z.gca().set_position((0.1, 0.20, 0.83, 0.70))
        if tail_selected == "Right" or tail_selected == "Both":
            z.plot(results["pos_α_vec"], label="Right tail")
            z.xlim(xmin=0.0, xmax=len(results["pos_α_vec"]) - 1)
        if tail_selected == "Left" or tail_selected == "Both":
            z.plot(results["neg_α_vec"], label="Left tail")
            z.xlim(xmin=0.0, xmax=len(results["neg_α_vec"]) - 1)

        z.ylabel(r"$\alpha$")
        z.title(
            "Time evolution of the parameter "
            + r"$\alpha$"
            + " for "
            + labels[i - 1]
            + "\n"
            + "Time period: "
            + dates[0]
            + " - "
            + dates[-1]
        )
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
        if tail_selected == "Right" or tail_selected == "Both":
            table_vals.append(
                [
                    "Right",
                    np.round(np.mean(results["pos_α_vec"]), 4),
                    np.round(np.median(results["pos_α_vec"]), 4),
                    np.round(np.std(results["pos_α_vec"]), 4),
                    np.round(np.min(results["pos_α_vec"]), 4),
                    np.round(np.max(results["pos_α_vec"]), 4),
                ]
            )
        if tail_selected == "Left" or tail_selected == "Both":
            table_vals.append(
                [
                    "Left",
                    np.round(np.mean(results["neg_α_vec"]), 4),
                    np.round(np.median(results["neg_α_vec"]), 4),
                    np.round(np.std(results["neg_α_vec"]), 4),
                    np.round(np.min(results["neg_α_vec"]), 4),
                    np.round(np.max(results["neg_α_vec"]), 4),
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
        # z.show()
        # z.show()

        # Plot the alpha exponent confidence interval in time
        if tail_selected == "Both" or tail_selected == "Right":

            z.figure("Time rolling CI for right tail for " + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["pos_α_vec"], color="green", label="Right tail")
            z.plot(results["pos_up_bound"],
                   color="purple", label="Upper bound")
            z.plot(results["pos_low_bound"], color="blue", label="Lower bound")
            z.plot(np.repeat(3, len(results["pos_α_vec"]) + 2), color="red")
            z.plot(np.repeat(2, len(results["pos_α_vec"]) + 2), color="red")
            z.ylabel(r"$\alpha$")
            z.xlim(xmin=0.0, xmax=len(results["pos_α_vec"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "Rolling confidence intervals for the "
                + r"$\alpha$"
                + "-right tail exponents "
                + "(c = "
                + str(1 - significance)
                + ")"
                + "\n"
                + "Ticker: "
                + labels[i - 1]
                + ".Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend(
                bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
                ncol=3,
                mode="expand",
                borderaxespad=0,
            )
            z.grid()
            # z.show()

            z.figure("Time rolling size for right tail for " + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["pos_abs_len"], color="green", label="Right tail")
            if tail_selected == "Both":
                z.plot(results["neg_abs_len"],
                       color="purple", label="Left tail")
            z.ylabel("Tail length")
            z.xlim(xmin=0.0, xmax=len(results["pos_abs_len"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "Rolling tail length for :"
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend()
            z.grid()
            # z.show()

            z.figure("Time rolling relative size for right tail for "
                     + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["pos_rel_len"], color="green", label="Right tail")
            if tail_selected == "Both":
                z.plot(results["neg_rel_len"],
                       color="purple", label="Left tail")
            z.ylabel("Relative tail length")
            z.xlim(xmin=0.0, xmax=len(results["pos_rel_len"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "Rolling relative tail length for :"
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend()
            z.grid()
            # z.show()

            z.figure("Time rolling KS test for right tail for " + labels[i-1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["pos_α_ks"], color="green", label="Right tail")
            if tail_selected == "Both":
                z.plot(results["neg_α_ks"], color="purple", label="Left tail")
            z.ylabel("p-value")
            z.xlim(xmin=0.0, xmax=len(results["pos_abs_len"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "KS-statistics: rolling p-value obtained from "
                "Clauset algorithm for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend()
            z.grid()
            # z.show()

            # Plotting the histograms for the rolling alpha

            z.figure(
                "Histogram of positive tail alphas for " + labels[i - 1],
                figsize=(8, 6),
                dpi=100,
            )
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            IQR = np.percentile(results["pos_α_vec"], 75) - np.percentile(
                results["pos_α_vec"], 25
            )
            h = 2 * IQR * np.power(len(results["pos_α_vec"]), -1.0 / 3.0)
            nbins = np.int(
                (np.max(results["pos_α_vec"]) - np.min(results["pos_α_vec"])) /
                float(h)
            )
            # Building the histogram and plotting the relevant vertical lines
            z.hist(results["pos_α_vec"], nbins, color="red")
            out1, bins = z.histogram(results["pos_α_vec"], nbins)
            z.plot(
                np.repeat(np.mean(results["pos_α_vec"]), np.max(out1) + 1),
                range(0, np.max(out1) + 1, 1),
                color="blue",
                linewidth=1.5,
                label=r"$E[\hat{\alpha}]$",
            )
            # Adding the labels, the axis limits, legend and grid
            z.xlabel(lab)
            z.ylabel("Absolute frequency")
            z.title(
                "Empirical distribution (right tail) of the rolling "
                + r"$\hat{\alpha}$"
                + " for "
                + labels[i - 1]
                + "\n"
                + "Time period: "
                + dates[0]
                + " - "
                + dates[-1]
            )
            z.xlim(xmin=np.min(results["pos_α_vec"]),
                   xmax=np.max(results["pos_α_vec"]))
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
                    np.round(np.mean(results["pos_α_vec"]), 4),
                    np.round(np.std(results["pos_α_vec"]), 4),
                    np.round(np.min(results["pos_α_vec"]), 4),
                    np.round(np.max(results["pos_α_vec"]), 4),
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
            # z.show()

        if tail_selected == "Both" or tail_selected == "Left":

            z.figure("Time rolling CI for left tail for " + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["neg_α_vec"], color="green", label="Right tail")
            z.plot(results["neg_up_bound"],
                   color="purple", label="Upper bound")
            z.plot(results["neg_low_bound"], color="blue", label="Lower bound")
            z.plot(np.repeat(3, len(results["neg_α_vec"]) + 2), color="red")
            z.plot(np.repeat(2, len(results["neg_α_vec"]) + 2), color="red")
            z.ylabel(r"$\alpha$")
            z.xlim(xmin=0.0, xmax=len(results["neg_α_vec"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "Rolling confidence intervals for the "
                + r"$\alpha$"
                + "-left tail exponents "
                + "(c = "
                + str(1 - significance)
                + ")"
                + "\n"
                + "Ticker: "
                + labels[i - 1]
                + ".Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend(
                bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
                ncol=3,
                mode="expand",
                borderaxespad=0,
            )
            z.grid()
            # z.show()

            z.figure("Time rolling size for left tail for " + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["neg_abs_len"], color="purple", label="Left tail")
            if tail_selected == "Both":
                z.plot(results["pos_abs_len"],
                       color="green", label="Right tail")
            z.ylabel("Tail length")
            z.xlim(xmin=0.0, xmax=len(results["neg_abs_len"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "Rolling tail length for :"
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend()
            z.grid()
            # z.show()

            z.figure("Time rolling relative size for left tail for " +
                     labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["neg_rel_len"], color="purple", label="Left tail")
            if tail_selected == "Both":
                z.plot(results["pos_rel_len"],
                       color="green", label="Right tail")
            z.ylabel("Relative tail length")
            z.xlim(xmin=0.0, xmax=len(results["neg_rel_len"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "Rolling relative tail length for :"
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend()
            z.grid()
            # z.show()

            z.figure("Time rolling KS test for left tail for " + labels[i - 1])
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            z.plot(results["neg_α_ks"], color="purple", label="Left tail")
            if tail_selected == "Both":
                z.plot(results["pos_α_ks"], color="green", label="Right tail")
            z.ylabel("p-value")
            z.xlim(xmin=0.0, xmax=len(results["neg_abs_len"]) - 1)
            z.xticks(
                range(0, len(spec_dates), spec_labelstep),
                [el[3:] for el in spec_dates[0::spec_labelstep]],
                rotation="vertical",
            )
            z.title(
                "KS-statistics: rolling p-value obtained "
                "from Clauset algorithm for "
                + labels[i - 1]
                + "\n"
                + "Time Period: "
                + dates[0]
                + " - "
                + dates[-1]
                + ". Input: "
                + lab
            )
            z.legend()
            z.grid()
            # z.show()

            # Plotting the histograms for the rolling alpha

            z.figure(
                "Histogram of negative tail alphas for " + labels[i - 1],
                figsize=(8, 6),
                dpi=100,
            )
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            IQR = np.percentile(results["neg_α_vec"], 75) - np.percentile(
                results["neg_α_vec"], 25
            )
            h = 2 * IQR * np.power(len(results["neg_α_vec"]), -1.0 / 3.0)
            nbins = np.int(
                (np.max(results["neg_α_vec"]) - np.min(results["neg_α_vec"])) /
                float(h)
            )
            # Building the histogram and plotting the relevant vertical lines
            z.hist(results["neg_α_vec"], nbins, color="red")
            out1, bins = z.histogram(results["neg_α_vec"], nbins)
            z.plot(
                np.repeat(np.mean(results["neg_α_vec"]), np.max(out1) + 1),
                range(0, np.max(out1) + 1, 1),
                color="blue",
                linewidth=1.5,
                label=r"$E[\hat{\alpha}]$",
            )
            # Adding the labels, the axis limits, legend and grid
            z.xlabel(lab)
            z.ylabel("Absolute frequency")
            z.title(
                "Empirical distribution (left tail) of the rolling "
                + r"$\hat{\alpha}$"
                + " for "
                + labels[i - 1]
                + "\n"
                + "Time period: "
                + dates[0]
                + " - "
                + dates[-1]
            )
            z.xlim(xmin=np.min(results["neg_α_vec"]),
                   xmax=np.max(results["neg_α_vec"]))
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
                    np.round(np.mean(results["neg_α_vec"]), 4),
                    np.round(np.std(results["neg_α_vec"]), 4),
                    np.round(np.min(results["neg_α_vec"]), 4),
                    np.round(np.max(results["neg_α_vec"]), 4),
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
            # z.show()

        # Print the figures
        matrix_form = np.array(tail_statistics)
        matrix_form_transpose = np.transpose(matrix_form)
        filename = ("TailStatistics_504_d=1_pn_normalized_" +
                    labels[i - 1] + "_KS.csv")
        df = pd.DataFrame(
            {
                "Date": spec_dates,
                "Positive Tail Exponent": matrix_form_transpose[0],
                "Negative Tail Exponent": matrix_form_transpose[1],
                "Positive Tail xmin": matrix_form_transpose[2],
                "Negative Tail xmin": matrix_form_transpose[3],
                "Positive Tail S.Err": matrix_form_transpose[4],
                "Negative Tail S.Err": matrix_form_transpose[5],
                "Positive Tail Size": matrix_form_transpose[6],
                "Negative Tail Size": matrix_form_transpose[7],
                "Positive Tail KS p-value": matrix_form_transpose[8],
                "Negative Tail KS p-value": matrix_form_transpose[9],
                "LL Ratio Right Tail TPL": matrix_form_transpose[10],
                "LL Ratio Right Tail Exp": matrix_form_transpose[11],
                "LL Ratio Right Tail LogN": matrix_form_transpose[12],
                "LL p-value Right Tail TPL": matrix_form_transpose[13],
                "LL p-value Right Tail Exp": matrix_form_transpose[14],
                "LL p-value Right Tail LogN": matrix_form_transpose[15],
                "LL Ratio Left Tail TPL": matrix_form_transpose[16],
                "LL Ratio Left Tail Exp": matrix_form_transpose[17],
                "LL Ratio Left Tail LogN": matrix_form_transpose[18],
                "LL p-value Left Tail TPL": matrix_form_transpose[19],
                "LL p-value Left Tail Exp": matrix_form_transpose[20],
                "LL p-value Left Tail LogN": matrix_form_transpose[21],
            }
        )

        df = df[
            [
                "Date",
                "Positive Tail Exponent",
                "Negative Tail Exponent",
                "Positive Tail xmin",
                "Negative Tail xmin",
                "Positive Tail S.Err",
                "Negative Tail S.Err",
                "Positive Tail Size",
                "Negative Tail Size",
                "Positive Tail KS p-value",
                "Negative Tail KS p-value",
                "LL Ratio Right Tail TPL",
                "LL Ratio Right Tail Exp",
                "LL Ratio Right Tail LogN",
                "LL p-value Right Tail TPL",
                "LL p-value Right Tail Exp",
                "LL p-value Right Tail LogN",
                "LL Ratio Left Tail TPL",
                "LL Ratio Left Tail Exp",
                "LL Ratio Left Tail LogN",
                "LL p-value Left Tail TPL",
                "LL p-value Left Tail Exp",
                "LL p-value Left Tail LogN",
            ]
        ]
        df.to_csv(filename, index=False)

    if tail_selected == "Right" or tail_selected == "Both":
        z.figure("Positive Power Law Boxplot")
        z.boxplot(positive_alpha_mat)
        z.xticks(range(1, len(labels) + 1, 1), labels)
        z.xlim(xmin=0.5, xmax=len(labels) + 0.5)
        z.ylabel(r"$\alpha$")
        z.title(
            "Boxplot representation of the "
            + r"$\alpha$"
            + "-right tail exponent "
            + "\n"
            + "Time Period: "
            + dates[0]
            + " - "
            + dates[-1]
            + ". Input series: "
            + lab
        )
        z.grid()
        # z.show()

    if tail_selected == "Left" or tail_selected == "Both":
        z.figure("Negative Power Law Boxplot")
        z.boxplot(negative_alpha_mat)
        z.xticks(range(1, len(labels) + 1, 1), labels)
        z.xlim(xmin=0.5, xmax=len(labels) + 0.5)
        z.ylabel(r"$\alpha$")
        z.title(
            "Boxplot representation of the "
            + r"$\alpha$"
            + "-left tail exponent"
            + "\n"
            + "Time Period: "
            + dates[0]
            + " - "
            + dates[-1]
            + ". Input series: "
            + lab
        )
        z.grid()
        # z.show()
