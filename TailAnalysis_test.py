#####################################
# Libraries                         #
#####################################
#  import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as z

from types import SimpleNamespace

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


# Common functions in both if-branches

def PowerLawFit(data, data_nature, xmin_rule, xmin_value, xmin_sign):

    data1 = np.array(data)[np.nonzero(data)]
    # data_nature is one of ['Discrete', 'Continuous']
    discrete = False if data_nature == 'Continuous' else True

    if xmin_rule == "Clauset":
        xmin = None
    elif xmin_rule == "Manual":
        xmin = xmin_value
    elif xmin_rule == "Percentile":
        xmin = np.percentile(data, xmin_sign)

    return pl.Fit(data1, discrete=discrete, xmin=xmin)


def fit_tail(tail_selected, tail_data):

    if tail_selected == "Right" or tail_selected == "Both":
        tail_plus = tail_data
        fit_right = PowerLawFit(tail_plus, data_nature,
                                xmin_rule, xmin_value, xmin_sign)
        # TODO: test standardization branch
        if standardize == "Yes" and standardize_target == "Tail":
            xmin = fit_right.power_law.xmin
            fit_right = standardize_tail(tail_plus, xmin)

    if tail_selected == "Left" or tail_selected == "Both":
        #  tail_neg = (np.dot(-1.0, tail_data)).tolist()
        tail_neg = -1 * tail_data
        fit_left = PowerLawFit(tail_neg, data_nature,
                               xmin_rule, xmin_value, xmin_sign)
        # TODO: test standardization branch
        if standardize == "Yes" and standardize_target == "Tail":
            xmin = fit_left.power_law.xmin
            fit_right = standardize_tail(tail_neg, xmin)

    return tail_plus, tail_neg, fit_right, fit_left


# TODO: test standardization function
def standardize_tail(tail_data, xmin):
    print("I am standardizing your tail")
    S = np.array(tail_data)
    S = S[S >= xmin]
    m = np.mean(S)
    v = np.std(S)
    X = (S - m) / v

    if abs_value == "Yes" and abs_target == "Tail":
        X = absolutize_tail(X)

    return PowerLawFit(X, data_nature, xmin_rule, np.min(X))


# TODO: test absolutize function
def absolutize_tail(tail_data):
    print("I am taking the absolute value of your tail")
    #  lab = "|" + lab + "|"
    return np.abs(tail_data)


def get_tail_stats(fit_obj, tail_data, ks_pvgof_tup):
    alpha = fit_obj.power_law.alpha
    xmin = fit_obj.power_law.xmin
    s_err = fit_obj.power_law.sigma
    tail_size = len(tail_data[tail_data >= xmin])
    ks_pv = ks_pvgof_tup[0]
    return alpha, xmin, s_err, tail_size, ks_pv


# NOTE: do the right/left values have to alternate in output CSV?
def tail_stat_zipper(tstat1, tstat2):
    return [val for pair in zip(tstat1, tstat2) for val in pair]


# TODO: merge this extraction function into get_tail_stats()
def get_logl_tstats(daily_log_ratio, daily_log_pv):
    r_tpl, r_exp, r_logn = daily_log_ratio
    p_tpl, p_exp, p_logn = daily_log_pv
    return list((r_tpl, r_exp, r_logn, p_tpl, p_exp, p_logn))


# TODO: factor plot making & storing code sections
# TODO: use config file (json, yaml, toml) for attr. (color, width, etc.)
# NOTES & IDEAS: create map (json) from plot data to its title, labels, etc.
# NOTE on refactor order: alpha-fit, time-rolling (4 sets), histogram, boxplot
# ASK: plots shown vs. stored are different -> why not store own plots too???


#####################################
# Script inputs (hardcoded for the time)                     #
#####################################

# TODO: wrap all user-input into an object,
#       and just pass that around as settings

database_name = "dbMSTR_test.csv"

no_entries = 1
fieldNames = ["# " + str(i) for i in range(1, no_entries + 1, 1)]
fieldValues = ["DE 01Y"]  # , "DE 03Y", "DE 05Y", "DE 10Y"]
labels = fieldValues

database = Extractor(database_name, labels)

# FIXME?: using date below -> ValueError: cannot convert float NaN to integer
#  initial_date = "2/5/2016"
initial_date = "1/1/2016"
final_date = "5/5/2016"
lookback = 504

input_type = "Log-Returns"

tau = 1

standardize = "No"
standardize_target = "Tail"  # choices is one of ['Full Series', 'Tail']

abs_value = "No"
abs_target = "Tail"  # choices is one of ['Full Series', 'Tail']

approach = "Rolling"  # choices is one of ['Static', 'Rolling', 'Increasing']
an_freq = 1
tail_selected = "Both"
data_nature = "Continuous"

xmin_rule = "Clauset"
xmin_value = None  # only used for xmin_rule == "Manual"
xmin_sign = None  # only used for xmin_rule == "Percentile"

if tail_selected == "Both":
    multiplier = 0.5
else:
    multiplier = 1.0

significance = 0.05

c_iter = 100


# NOTE: these lists appear to only be used for plotting
def results_lists_init():
    labels = ("pos_α_vec", "neg_α_vec", "pos_α_ks", "neg_α_ks",
              "pos_up_bound", "neg_up_bound", "pos_low_bound", "neg_low_bound",
              "pos_abs_len", "neg_abs_len", "pos_rel_len", "neg_rel_len",
              "loglr_right", "loglr_left", "loglpv_right", "loglpv_left")
    # NOTE: length of each list is the number of days -> so use np.ndarray
    return {l: [] for l in labels}


# INITIALIZE non-user specified global variables

# lists to store the results for plotting (16 total)
results = results_lists_init()

initial_index = database[0].index(initial_date)
final_index = database[0].index(final_date)
dates = database[0][initial_index: (final_index + 1)]
labelstep = (22 if len(dates) <= 252 else
             66 if (len(dates) > 252 and len(dates) <= 756) else
             121)
N = len(database)


# TODO: set values of these dynamically based on user input
# object to hold all options data determined by user input data
options_data = {"standardize": False,
                "absolutize": False,
                "approach": "Rolling",
                "analysis_freq": 1,
                "use_right_tail": (True if tail_selected in ["Right", "Both"]
                                   else False),
                "use_left_tail": (True if tail_selected in ["Left", "Both"]
                                  else False),
                "data_nature": "Continuous",
                "xmin_rule": "Clauset",
                "dates": dates,
                "labelstep": labelstep}
# TODO: add "labels" and other important values into options dict
options = SimpleNamespace(**options_data)


# Execution logic for the actual calculations

if approach == "Static":

    # TODO: add list below to results_lists_init function?
    tail_statistics = []

    for i in range(1, N, 1):

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

        # TODO: replace below with standardize_tail() but for "Full Series"
        #  if standardize == "Yes":
        #      if standardize_target == "Full Series":
        #          print("I am standardizing your time series")
        #          S = X
        #          m = np.mean(S)
        #          v = np.std(S)
        #          X = (S - m) / v

        # TODO: replace below with absolutize_tail()
        #  if abs_value == "Yes":
        #      if abs_target == "Full Series":
        #          print("I am taking the absolute value of your time series")
        #          X = np.abs(X)
        #          lab = "|" + lab + "|"

        tail_plus, tail_neg, fit_1, fit_2 = fit_tail(tail_selected, X)

        # TODO: when only Right or Left tail selected,
        #       the other fit object will be None
        alpha1 = fit_1.power_law.alpha
        xmin1 = fit_1.power_law.xmin
        s_err1 = fit_1.power_law.sigma

        alpha2 = fit_2.power_law.alpha
        xmin2 = fit_2.power_law.xmin
        s_err2 = fit_2.power_law.sigma

        if tail_selected == "Right" or tail_selected == "Both":
            p1 = plpva.plpva(tail_plus, float(xmin1), "reps", c_iter, "silent")
            results["pos_α_ks"].append(p1[0])

        if tail_selected == "Left" or tail_selected == "Both":
            p2 = plpva.plpva(tail_neg, float(xmin2), "reps", c_iter, "silent")
            results["neg_α_ks"].append(p2[0])

        # Figures Plot & Show Sections below
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

            results["pos_α_vec"].append(alpha1)
            results["pos_up_bound"].append(
                alpha1 + (st.norm.ppf(1 - multiplier * significance)) * s_err1)
            results["pos_low_bound"].append(
                alpha1 - (st.norm.ppf(1 - multiplier * significance)) * s_err1)
            results["pos_abs_len"].append(len(filter(lambda x: x >= xmin1,
                                                     tail_plus)))
            results["pos_rel_len"].append(
                len(filter(lambda x: x >= xmin1, tail_plus)) /
                float(len(tail_plus)))

        if tail_selected == "Left" or tail_selected == "Both":
            results["neg_α_vec"].append(alpha2)
            results["neg_up_bound"].append(
                alpha2 + (st.norm.ppf(1 - multiplier * significance)) * s_err2)
            results["neg_low_bound"].append(
                alpha2 - (st.norm.ppf(1 - multiplier * significance)) * s_err2)
            results["neg_abs_len"].append(len(filter(lambda x: x >= xmin2,
                                                     tail_neg)))
            results["neg_rel_len"].append(
                len(filter(lambda x: x >= xmin2, tail_neg)) /
                float(len(tail_neg)))

        # Building tail statistics section

        if tail_selected == "Right" or tail_selected == "Both":
            tstat_right = get_tail_stats(fit_1, tail_plus, p1)
        if tail_selected == "Left" or tail_selected == "Both":
            tstat_left = get_tail_stats(fit_2, tail_neg, p2)

        if tail_selected == "Both":
            row = tail_stat_zipper(tstat_right, tstat_left)
        elif tail_selected == "Right":
            row = tail_stat_zipper(tstat_right, np.zeros(len(tstat_right)))
        elif tail_selected == "Left":
            row = tail_stat_zipper(np.zeros(len(tstat_left)), tstat_left)

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

    # Write Tail Statistics to CSV file
    filename = "TailStatistics_Overall.csv"
    labels_colvec = np.array(labels).reshape(len(labels), 1)
    df_data = np.hstack((labels_colvec, tail_statistics))
    column_headers = ["Input",
                      "Positive Tail Exponent",
                      "Negative Tail Exponent",
                      "Positive Tail xmin",
                      "Negative Tail xmin",
                      "Positive Tail S.Err",
                      "Negative Tail S.Err",
                      "Positive Tail Size",
                      "Negative Tail Size",
                      "Positive Tail KS p-value",
                      "Negative Tail KS p-value"]
    df = pd.DataFrame(df_data, columns=column_headers)
    df.to_csv(filename, index=False)

elif approach == "Rolling" or approach == "Increasing":

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

    temp = []  # NOTE: appears to be unused

    if an_freq > 1:
        spec_dates = []
        for ddd in range(0, len(dates), an_freq):
            spec_dates.append(dates[ddd])
        spec_labelstep = 22
    else:
        spec_dates = dates
        spec_labelstep = labelstep

    # int(np.maximum(np.floor(22/float(an_freq)),1.0))

    # TODO: add lists below to results_lists_init function?
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

            tail_plus, tail_neg, fit_1, fit_2 = fit_tail(tail_selected, X)

            # TODO: when only Right or Left tail selected,
            #       the other fit object will be None
            alpha1 = fit_1.power_law.alpha
            xmin1 = fit_1.power_law.xmin
            s_err1 = fit_1.power_law.sigma
            alpha2 = fit_2.power_law.alpha
            xmin2 = fit_2.power_law.xmin
            s_err2 = fit_2.power_law.sigma

            #  # Plot Storing if-block
            #  if plot_storing == "Yes":
            #
            #      if tail_selected == "Right" or tail_selected == "Both":
            #
            #          plt.figure(
            #              "Right tail scaling for "
            #              + labels[i - 1]
            #              + begin_date
            #              + "_"
            #              + end_date
            #          )
            #          z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            #          fig4 = fit_1.plot_ccdf(
            #              color="b", linewidth=2, label="Empirical CCDF"
            #          )
            #          fit_1.power_law.plot_ccdf(
            #              color="b", linestyle="-", label="Fitted CCDF", ax=fig4
            #          )
            #          fit_1.plot_pdf(
            #              color="r", linewidth=2, label="Empirical PDF", ax=fig4
            #          )
            #          fit_1.power_law.plot_pdf(
            #              color="r", linestyle="-", label="Fitted PDF", ax=fig4
            #          )
            #          fig4.set_title(
            #              "Log-log plot of the scaling properties "
            #              "of the right-tail for "
            #              + labels[i - 1]
            #              + "\n"
            #              + "Time Period: "
            #              + begin_date
            #              + " - "
            #              + end_date
            #              + ". Input series: "
            #              + lab
            #          )
            #          fig4.grid()
            #          fig4.legend()
            #          col_labels = [
            #              r"$\hat{\alpha}$",
            #              "Standard err.",
            #              r"$x_{min}$",
            #              "size",
            #          ]
            #          table_vals = []
            #          table_vals.append(
            #              [
            #                  np.round(alpha1, 4),
            #                  np.round(s_err1, 4),
            #                  np.round(xmin1, 4),
            #                  len(filter(lambda x: x >= xmin1, tail_plus)),
            #              ]
            #          )
            #          the_table = plt.table(
            #              cellText=table_vals,
            #              cellLoc="center",
            #              colLabels=col_labels,
            #              loc="bottom",
            #              bbox=[0.0, -0.26, 1.0, 0.10],
            #          )
            #          the_table.auto_set_font_size(False)
            #          the_table.set_fontsize(10)
            #          the_table.scale(0.5, 0.5)
            #          plt.savefig(
            #              "Right-tail scaling_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #          plt.figure("Right tail comparison for " + labels[i - 1])
            #          fig4 = fit_1.plot_ccdf(
            #              color="b", linewidth=2, label="Empirical CCDF"
            #          )
            #          fit_1.power_law.plot_ccdf(
            #              color="r", linestyle="-", label="Fitted PL", ax=fig4
            #          )
            #          fit_1.truncated_power_law.plot_ccdf(
            #              color="g", linestyle="-", label="Fitted TPL", ax=fig4
            #          )
            #          fit_1.exponential.plot_ccdf(
            #              color="c", linestyle="-", label="Fitted Exp.", ax=fig4
            #          )
            #          fit_1.lognormal.plot_ccdf(
            #              color="m", linestyle="-", label="Fitted LogN.", ax=fig4
            #          )
            #          fig4.set_title(
            #              "Comparison of the distributions "
            #              "fitted on the right-tail for "
            #              + labels[i - 1]
            #              + "\n"
            #              + "Time Period: "
            #              + dates[0]
            #              + " - "
            #              + dates[-1]
            #              + ". Input series: "
            #              + lab
            #          )
            #          fig4.grid()
            #          fig4.legend()
            #          plt.savefig(
            #              "Right-tail fitting comparison_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #      if tail_selected == "Left" or tail_selected == "Both":
            #
            #          plt.figure(
            #              "Left tail scaling for "
            #              + labels[i - 1]
            #              + begin_date
            #              + "_"
            #              + end_date
            #          )
            #          z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            #          fig4 = fit_2.plot_ccdf(
            #              color="b", linewidth=2, label="Empirical CCDF"
            #          )
            #          fit_2.power_law.plot_ccdf(
            #              color="b", linestyle="-", label="Fitted CCDF", ax=fig4
            #          )
            #          fit_2.plot_pdf(
            #              color="r", linewidth=2, label="Empirical PDF", ax=fig4
            #          )
            #          fit_2.power_law.plot_pdf(
            #              color="r", linestyle="-", label="Fitted PDF", ax=fig4
            #          )
            #          fig4.set_title(
            #              "Log-log plot of the scaling properties "
            #              "of the left-tail for "
            #              + labels[i - 1]
            #              + "\n"
            #              + "Time Period: "
            #              + begin_date
            #              + " - "
            #              + end_date
            #              + ". Input series: "
            #              + lab
            #          )
            #          fig4.grid()
            #          fig4.legend()
            #          col_labels = [
            #              r"$\hat{\alpha}$",
            #              "Standard err.",
            #              r"$x_{min}$",
            #              "size",
            #          ]
            #          table_vals = []
            #          table_vals.append(
            #              [
            #                  np.round(alpha2, 4),
            #                  np.round(s_err2, 4),
            #                  np.round(xmin2, 4),
            #                  len(filter(lambda x: x >= xmin2, tail_neg)),
            #              ]
            #          )
            #          the_table = plt.table(
            #              cellText=table_vals,
            #              cellLoc="center",
            #              colLabels=col_labels,
            #              loc="bottom",
            #              bbox=[0.0, -0.26, 1.0, 0.10],
            #          )
            #          the_table.auto_set_font_size(False)
            #          the_table.set_fontsize(10)
            #          the_table.scale(0.5, 0.5)
            #          plt.savefig(
            #              "Left-tail scaling_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #          plt.figure("Left tail comparison for " + labels[i - 1])
            #          fig4 = fit_2.plot_ccdf(
            #              color="b", linewidth=2, label="Empirical CCDF"
            #          )
            #          fit_2.power_law.plot_ccdf(
            #              color="r", linestyle="-", label="Fitted PL", ax=fig4
            #          )
            #          fit_2.truncated_power_law.plot_ccdf(
            #              color="g", linestyle="-", label="Fitted TPL", ax=fig4
            #          )
            #          fit_2.exponential.plot_ccdf(
            #              color="c", linestyle="-", label="Fitted Exp.", ax=fig4
            #          )
            #          fit_2.lognormal.plot_ccdf(
            #              color="m", linestyle="-", label="Fitted LogN.", ax=fig4
            #          )
            #          fig4.set_title(
            #              "Comparison of the distributions fitted "
            #              "on the left-tail for "
            #              + labels[i - 1]
            #              + "\n"
            #              + "Time Period: "
            #              + dates[0]
            #              + " - "
            #              + dates[-1]
            #              + ". Input series: "
            #              + lab
            #          )
            #          fig4.grid()
            #          fig4.legend()
            #          plt.savefig(
            #              "Left-tail fitting comparison_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()

            if tail_selected == "Right" or tail_selected == "Both":

                results["pos_α_vec"].append(alpha1)
                results["pos_up_bound"].append(
                    alpha1 + (st.norm.ppf(1 - multiplier * significance))
                    * s_err1)
                results["pos_low_bound"].append(
                    alpha1 - (st.norm.ppf(1 - multiplier * significance))
                    * s_err1
                )
                results["pos_abs_len"].append(len(
                    tail_plus[tail_plus >= xmin1]))
                results["pos_rel_len"].append(
                    len(tail_plus[tail_plus >= xmin1]) /
                    float(len(tail_plus)))
                p1 = plpva.plpva(tail_plus, float(xmin1),
                                 "reps", c_iter, "silent")
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
                # NOTE: tail_plus was already converted;
                #       tail_neg now should be a np.ndarray by default
                #  tail_neg = np.array(tail_neg)
                results["neg_abs_len"].append(len(tail_neg[tail_neg >= xmin2]))
                results["neg_rel_len"].append(
                    len(tail_neg[tail_neg >= xmin2]) /
                    float(len(tail_neg)))
                p2 = plpva.plpva(tail_neg, float(xmin2),
                                 "reps", c_iter, "silent")
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

            # Building tail statistics section
            if tail_selected == "Right" or tail_selected == "Both":
                tstat_right = get_tail_stats(fit_1, tail_plus, p1)
                logl_tstat_right = get_logl_tstats(daily_r_ratio, daily_r_p)
            if tail_selected == "Left" or tail_selected == "Both":
                tstat_left = get_tail_stats(fit_2, tail_neg, p2)
                logl_tstat_left = get_logl_tstats(daily_l_ratio, daily_l_p)

            if tail_selected == "Both":
                row = (tail_stat_zipper(tstat_right, tstat_left) +
                       logl_tstat_right +
                       logl_tstat_left)
            elif tail_selected == "Right":
                row = (tail_stat_zipper(tstat_right,
                                        np.zeros(len(tstat_right)))
                       + logl_tstat_right
                       + list("0" * 6))
            elif tail_selected == "Left":
                row = (tail_stat_zipper(np.zeros(len(tstat_left)),
                                        tstat_left)
                       + list("0" * 6)
                       + logl_tstat_left)

            tail_statistics.append(row)

        # NOTE: these are used for the boxplots --> beware w/ multiprocessing
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

        # Write Tail Statistics to CSV file
        filename = ("TailStatistics_504_d=1_pn_normalized_" +
                    labels[i - 1] + "_KS.csv")
        date_colvec = np.array(spec_dates).reshape(len(spec_dates), 1)
        df_data = np.hstack((date_colvec, tail_statistics))
        column_headers = ["Date",
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
                          "LL p-value Left Tail LogN"]
        df = pd.DataFrame(df_data, columns=column_headers)
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
