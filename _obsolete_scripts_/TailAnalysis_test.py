from utils import io, structs, calc

import plot_funcs.tail_risk_plotter as trp
#  import plot_funcs.boxplot as pfbx


# TODO: factor plot making & storing code sections
# TODO: use config file (json, yaml, toml) for attr. (color, width, etc.)
# NOTES & IDEAS: create map (json) from plot data to its title, labels, etc.
# NOTE on refactor order: alpha-fit, time-rolling (4 sets), histogram, boxplot
# ASK: plots shown vs. stored are different -> why not store own plots too???


# Execution logic for the actual calculations

if s.approach == "static":

    # init 2D-array containing results to be written to CSV (per ticker)
    #  results_array = structs.init_csv_array(N=len(s.tickers), M=5*len(s.tails_used))
    results_array = structs.init_csv_array(N=len(s.tickers))
    # TODO: confirm M-dim of csv_array for static approach (how many stats)

    # NOTE: logl_stats only used for barplots
    # TODO: vecs for storing logl_stats not initialized in csv_array above

    #  for i in range(1, N, 1):
    for t, tck in enumerate(s.tickers):

        print(f"I am analyzing the time series for {tck} "
              f"between {s.date_i} and {s.date_f}")
        #  series = database[i][initial_index: (final_index + 1)]
        series = s.db_df[tck].iloc[s.ind_i: s.ind_f+1].array

        # TODO: for static approach, logl_stats only needed for barplots
        results_array[t, :] = calc.get_results_tup(series)

        #  csv_data = results_array[:5*len(s.tails_used)]
        #  logl_stats = results_array[5*len(s.tails_used):]  # NOTE: only plts

        #  # Figures Plot & Show Sections below
        #  if tail_selected == "right" or tail_selected == "both":
        #
        #      plt.figure("right tail scaling for " + tickers[i - 1])
        #      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
        #      fig4 = fit_1.plot_ccdf(color="b", linewidth=2,
        #                             label="Empirical CCDF")
        #      fit_1.power_law.plot_ccdf(
        #          color="b", linestyle="-", label="Fitted CCDF", ax=fig4
        #      )
        #      fit_1.plot_pdf(color="r", linewidth=2,
        #                     label="Empirical PDF", ax=fig4)
        #      fit_1.power_law.plot_pdf(
        #          color="r", linestyle="-", label="Fitted PDF", ax=fig4
        #      )
        #      fig4.set_title(
        #          "Log-log plot of the scaling properties of the right-tail for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      fig4.grid()
        #      fig4.legend()
        #      col_labels = [r"$\hat{\alpha}$", "Standard err.",
        #                    r"$x_{min}$", "size"]
        #      table_vals = []
        #      table_vals.append(
        #          [
        #              np.round(alpha1, 4),
        #              np.round(s_err1, 4),
        #              np.round(xmin1, 4),
        #              len(filter(lambda x: x > xmin1, tail_plus)),
        #          ]
        #      )
        #      the_table = plt.table(
        #          cellText=table_vals,
        #          cellLoc="center",
        #          colLabels=col_labels,
        #          loc="bottom",
        #          bbox=[0.0, -0.26, 1.0, 0.10],
        #      )
        #      the_table.auto_set_font_size(False)
        #      the_table.set_fontsize(10)
        #      the_table.scale(0.5, 0.5)
        #      plt.show()
        #
        #      plt.figure("right tail comparison for " + tickers[i - 1])
        #      fig4 = fit_1.plot_ccdf(color="b", linewidth=2,
        #                             label="Empirical CCDF")
        #      fit_1.power_law.plot_ccdf(
        #          color="r", linestyle="-", label="Fitted PL", ax=fig4
        #      )
        #      fit_1.truncated_power_law.plot_ccdf(
        #          color="g", linestyle="-", label="Fitted TPL", ax=fig4
        #      )
        #      fit_1.exponential.plot_ccdf(
        #          color="c", linestyle="-", label="Fitted Exp.", ax=fig4
        #      )
        #      fit_1.lognormal.plot_ccdf(
        #          color="m", linestyle="-", label="Fitted LogN.", ax=fig4
        #      )
        #      fig4.set_title(
        #          "Comparison of the distributions fitted on the right-tail for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      fig4.grid()
        #      fig4.legend()
        #      plt.show()
        #
        #      distribution_list = ["truncated_power_law",
        #                           "exponential", "lognormal"]
        #      for pdf in distribution_list:
        #          R, p = fit_1.distribution_compare(
        #              "power_law", pdf, normalized_ratio=True)
        #          #  loglikelihood_ratio_right.append(R)
        #          results["loglr_right"].append(R)
        #          #  loglikelihood_pvalue_right.append(p)
        #          results["loglpv_right"].append(p)
        #
        #      z.figure("Log Likelihood ratio for the right tail for " +
        #               tickers[i - 1])
        #      #  z.bar(
        #      #      np.arange(0, len(loglikelihood_ratio_right), 1),
        #      #      loglikelihood_ratio_right, 1,)
        #      z.bar(
        #          np.arange(0, len(results["loglr_right"]), 1),
        #          results["loglr_right"], 1,)
        #      z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
        #               distribution_list)
        #      z.ylabel("R")
        #      z.title(
        #          "Log-likelihood ratio for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      z.grid()
        #      # z.show()
        #
        #      z.figure("Log Likelihood ratio p-values for the right tail for " +
        #               tickers[i - 1])
        #      #  z.bar(
        #      #      np.arange(0, len(loglikelihood_pvalue_right), 1),
        #      #      loglikelihood_pvalue_right, 1,)
        #      z.bar(
        #          np.arange(0, len(results["loglpv_right"]), 1),
        #          results["loglpv_right"], 1,)
        #      z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
        #               distribution_list)
        #      z.ylabel("R")
        #      z.title(
        #          "Log-likelihood ratio p values for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      z.grid()
        #      # z.show()
        #
        #  if tail_selected == "left" or tail_selected == "both":
        #
        #      plt.figure("left tail scaling for " + tickers[i - 1])
        #      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
        #      fig4 = fit_2.plot_ccdf(color="b", linewidth=2,
        #                             label="Empirical CCDF")
        #      fit_2.power_law.plot_ccdf(
        #          color="b", linestyle="-", label="Fitted CCDF", ax=fig4
        #      )
        #      fit_2.plot_pdf(color="r", linewidth=2,
        #                     label="Empirical PDF", ax=fig4)
        #      fit_2.power_law.plot_pdf(
        #          color="r", linestyle="-", label="Fitted PDF", ax=fig4
        #      )
        #      fig4.set_title(
        #          "Log-log plot of the scaling properties of the left-tail for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      fig4.grid()
        #      fig4.legend()
        #      col_labels = [r"$\hat{\alpha}$", "Standard err.",
        #                    r"$x_{min}$", "size"]
        #      table_vals = []
        #      table_vals.append(
        #          [
        #              np.round(alpha2, 4),
        #              np.round(s_err2, 4),
        #              np.round(xmin2, 4),
        #              len(filter(lambda x: x > xmin2, tail_neg)),
        #          ]
        #      )
        #      the_table = plt.table(
        #          cellText=table_vals,
        #          cellLoc="center",
        #          colLabels=col_labels,
        #          loc="bottom",
        #          bbox=[0.0, -0.26, 1.0, 0.10],
        #      )
        #      the_table.auto_set_font_size(False)
        #      the_table.set_fontsize(10)
        #      the_table.scale(0.5, 0.5)
        #      plt.show()
        #
        #      plt.figure("left tail comparison for " + tickers[i - 1])
        #      fig4 = fit_2.plot_ccdf(color="b", linewidth=2,
        #                             label="Empirical CCDF")
        #      fit_2.power_law.plot_ccdf(
        #          color="r", linestyle="-", label="Fitted PL", ax=fig4
        #      )
        #      fit_2.truncated_power_law.plot_ccdf(
        #          color="g", linestyle="-", label="Fitted TPL", ax=fig4
        #      )
        #      fit_2.exponential.plot_ccdf(
        #          color="c", linestyle="-", label="Fitted Exp.", ax=fig4
        #      )
        #      fit_2.lognormal.plot_ccdf(
        #          color="m", linestyle="-", label="Fitted LogN.", ax=fig4
        #      )
        #      fig4.set_title(
        #          "Comparison of the distributions fitted on the left-tail for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      fig4.grid()
        #      fig4.legend()
        #      plt.show()
        #
        #      distribution_list = ["truncated_power_law",
        #                           "exponential", "lognormal"]
        #      for pdf in distribution_list:
        #          R, p = fit_2.distribution_compare(
        #              "power_law", pdf, normalized_ratio=True
        #          )
        #          #  loglikelihood_ratio_left.append(R)
        #          results["loglr_left"].append(R)
        #          #  loglikelihood_pvalue_left.append(p)
        #          results["loglpv_left"].append(p)
        #
        #      z.figure("Log Likelihood ratio for the left tail for " +
        #               tickers[i - 1])
        #      #  z.bar(
        #      #      np.arange(0, len(loglikelihood_ratio_left), 1),
        #      #      loglikelihood_ratio_left, 1,)
        #      z.bar(
        #          np.arange(0, len(results["loglr_left"]), 1),
        #          results["loglr_left"], 1,)
        #      z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
        #               distribution_list)
        #      z.ylabel("R")
        #      z.title(
        #          "Log-likelihood ratio for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      z.grid()
        #      # z.show()
        #
        #      z.figure(
        #          "Log Likelihood ratio p-values for the left tail for " +
        #          tickers[i - 1])
        #      #  z.bar(
        #      #      np.arange(0, len(loglikelihood_pvalue_left), 1),
        #      #      loglikelihood_pvalue_left, 1,)
        #      z.bar(
        #          np.arange(0, len(results["loglpv_left"]), 1),
        #          results["loglpv_left"], 1,)
        #      z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
        #               distribution_list)
        #      z.ylabel("R")
        #      z.title(
        #          "Log-likelihood ratio p values for "
        #          + tickers[i - 1]
        #          + "\n"
        #          + "Time Period: "
        #          + dates[0]
        #          + " - "
        #          + dates[-1]
        #          + ". Input series: "
        #          + lab
        #      )
        #      z.grid()
        #      # z.show()

    #  # Preparing the figure
    #
    #  z.figure("static alpha")
    #  z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    #  if tail_selected == "right" or tail_selected == "both":
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  positive_alpha_vec,
    #          results["pos_α_vec"],
    #          marker="^",
    #          markersize=10.0,
    #          linewidth=0.0,
    #          color="green",
    #          label="right tail",
    #      )
    #  if tail_selected == "left" or tail_selected == "both":
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  negative_alpha_vec,
    #          results["neg_α_vec"],
    #          marker="^",
    #          markersize=10.0,
    #          linewidth=0.0,
    #          color="red",
    #          label="left tail",
    #      )
    #  z.xticks(range(1, len(tickers) + 1, 1), tickers)
    #  z.xlim(xmin=0.5, xmax=len(tickers) + 0.5)
    #  z.ylabel(r"$\alpha$")
    #  z.title(
    #      "Estimation of the "
    #      + r"$\alpha$"
    #      + "-right tail exponents using KS-Method"
    #      + "\n"
    #      + "Time Period: "
    #      + dates[0]
    #      + " - "
    #      + dates[-1]
    #      + ". Input series: "
    #      + lab
    #  )
    #  z.legend(
    #      bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=2,
    #      mode="expand", borderaxespad=0
    #  )
    #  z.grid()
    #  # z.show()
    #
    #  if tail_selected == "right" or tail_selected == "both":
    #
    #      # Confidence interval for the right tail
    #      z.figure("Confidence interval for the right tail")
    #      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  positive_alpha_vec,
    #          results["pos_α_vec"],
    #          marker="o",
    #          markersize=7.0,
    #          linewidth=0.0,
    #          color="green",
    #          label="right tail",
    #      )
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  positive_upper_bound,
    #          results["pos_up_bound"],
    #          marker="o",
    #          markersize=7.0,
    #          linewidth=0.0,
    #          color="purple",
    #          label="Upper bound",
    #      )
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  positive_lower_bound,
    #          results["pos_low_bound"],
    #          marker="o",
    #          markersize=7.0,
    #          linewidth=0.0,
    #          color="blue",
    #          label="Lower bound",
    #      )
    #      z.plot(
    #          range(0, len(tickers) + 2, 1), np.repeat(3, len(tickers) + 2),
    #          color="red"
    #      )
    #      z.plot(
    #          range(0, len(tickers) + 2, 1), np.repeat(2, len(tickers) + 2),
    #          color="red"
    #      )
    #      z.xticks(range(1, len(tickers) + 1, 1), tickers)
    #      z.xlim(xmin=0.5, xmax=len(tickers) + 0.5)
    #      z.ylabel(r"$\alpha$")
    #      z.title(
    #          "Confidence intervals for the "
    #          + r"$\alpha$"
    #          + "-right tail exponents "
    #          + "(c = "
    #          + str(1 - significance)
    #          + ")"
    #          + "\n"
    #          + "Time Period: "
    #          + dates[0]
    #          + " - "
    #          + dates[-1]
    #          + ". Input series: "
    #          + lab
    #      )
    #      z.legend(
    #          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
    #          ncol=3,
    #          mode="expand",
    #          borderaxespad=0,
    #      )
    #      z.grid()
    #      # z.show()
    #
    #  if tail_selected == "left" or tail_selected == "both":
    #
    #      # Confidence interval for the left tail
    #      z.figure("Confidence interval for the left tail")
    #      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  negative_alpha_vec,
    #          results["neg_α_vec"],
    #          marker="o",
    #          markersize=7.0,
    #          linewidth=0.0,
    #          color="green",
    #          label="left tail",
    #      )
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  negative_upper_bound,
    #          results["neg_up_bound"],
    #          marker="o",
    #          markersize=7.0,
    #          linewidth=0.0,
    #          color="purple",
    #          label="Upper bound",
    #      )
    #      z.plot(
    #          range(1, len(tickers) + 1, 1),
    #          #  negative_lower_bound,
    #          results["neg_low_bound"],
    #          marker="o",
    #          markersize=7.0,
    #          linewidth=0.0,
    #          color="blue",
    #          label="Lower bound",
    #      )
    #      z.plot(
    #          range(0, len(tickers) + 2, 1), np.repeat(3, len(tickers) + 2),
    #          color="red"
    #      )
    #      z.plot(
    #          range(0, len(tickers) + 2, 1), np.repeat(2, len(tickers) + 2),
    #          color="red"
    #      )
    #      z.xticks(range(1, len(tickers) + 1, 1), tickers)
    #      z.xlim(xmin=0.5, xmax=len(tickers) + 0.5)
    #      z.ylabel(r"$\alpha$")
    #      z.title(
    #          "Confidence intervals for the "
    #          + r"$\alpha$"
    #          + "-left tail exponents "
    #          + "(c = "
    #          + str(1 - significance)
    #          + ")"
    #          + "\n"
    #          + "Time Period: "
    #          + dates[0]
    #          + " - "
    #          + dates[-1]
    #          + ". Input series: "
    #          + lab
    #      )
    #      z.legend(
    #          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
    #          ncol=3,
    #          mode="expand",
    #          borderaxespad=0,
    #      )
    #      z.grid()
    #      # z.show()
    #
    #  # Absolute length of the tail bar chart
    #
    #  z.figure("Absolute tail lengths")
    #  z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    #  amplitude = 0.5
    #  if tail_selected == "right" or tail_selected == "both":
    #      z.bar(
    #          np.arange(0, 2 * len(tickers), 2),
    #          #  positive_abs_length,
    #          results["pos_abs_len"],
    #          amplitude,
    #          color="green",
    #          label="right tail",
    #      )
    #  if tail_selected == "left" or tail_selected == "both":
    #      z.bar(
    #          np.arange(amplitude, 2 * len(tickers) + amplitude, 2),
    #          #  negative_abs_length,
    #          results["neg_abs_len"],
    #          amplitude,
    #          color="red",
    #          label="left tail",
    #      )
    #  z.xticks(np.arange(amplitude, 2 * len(tickers) + amplitude, 2), tickers)
    #  z.ylabel("Tail length")
    #  z.title(
    #      "Bar chart representation of the length of the tails"
    #      + "\n"
    #      + "Time Period: "
    #      + dates[0]
    #      + " - "
    #      + dates[-1]
    #      + ". Input series: "
    #      + lab
    #  )
    #  z.legend(
    #      bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3, mode="expand",
    #      borderaxespad=0
    #  )
    #  z.grid()
    #  # z.show()
    #
    #  # Absolute length of the tail bar chart
    #
    #  z.figure("Relative tail lengths")
    #  z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    #  amplitude = 0.5
    #  if tail_selected == "right" or tail_selected == "both":
    #      z.bar(
    #          np.arange(0, 2 * len(tickers), 2),
    #          #  positive_rel_length,
    #          results["pos_rel_len"],
    #          amplitude,
    #          color="green",
    #          label="right tail",
    #      )
    #  if tail_selected == "left" or tail_selected == "both":
    #      z.bar(
    #          np.arange(amplitude, 2 * len(tickers) + amplitude, 2),
    #          #  negative_rel_length,
    #          results["neg_rel_len"],
    #          amplitude,
    #          color="red",
    #          label="left tail",
    #      )
    #  z.xticks(np.arange(amplitude, 2 * len(tickers) + amplitude, 2), tickers)
    #  z.ylabel("Tail relative length")
    #  z.title(
    #      "Bar chart representation of the relative length of the tails"
    #      + "\n"
    #      + "Time Period: "
    #      + dates[0]
    #      + " - "
    #      + dates[-1]
    #      + ". Input series: "
    #      + lab
    #  )
    #  z.legend(
    #      bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3,
    #      mode="expand", borderaxespad=0
    #  )
    #  z.grid()
    #  # z.show()
    #
    #  # KS test outcome
    #
    #  z.figure("KS test p value for the tails")
    #  z.gca().set_position((0.1, 0.20, 0.83, 0.70))
    #  amplitude = 0.5
    #  if tail_selected == "right" or tail_selected == "both":
    #      z.bar(
    #          np.arange(0, 2 * len(tickers), 2),
    #          #  positive_alpha_KS,
    #          results["pos_α_ks"],
    #          amplitude,
    #          color="green",
    #          label="right tail",
    #      )
    #  if tail_selected == "left" or tail_selected == "both":
    #      z.bar(
    #          np.arange(amplitude, 2 * len(tickers) + amplitude, 2),
    #          #  negative_alpha_KS,
    #          results["neg_α_ks"],
    #          amplitude,
    #          color="red",
    #          label="left tail",
    #      )
    #  z.xticks(np.arange(amplitude, 2 * len(tickers) + amplitude, 2), tickers)
    #  z.ylabel("p-value")
    #  z.title(
    #      "KS-statistics: p-value obtained from Clauset algorithm"
    #      + "\n"
    #      + "Time Period: "
    #      + dates[0]
    #      + " - "
    #      + dates[-1]
    #      + ". Input series: "
    #      + lab
    #  )
    #  z.legend(
    #      bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3,
    #      mode="expand", borderaxespad=0
    #  )
    #  z.grid()
    #  # z.show()

    #  csv_data = results_array[:, :5*len(s.tails_used)]  # NOTE: == tail_stats
    #  import numpy as np
    #  import pandas as pd
    #  # Write Tail Statistics to CSV file
    #  filename = "TailStatistics_Overall.csv"
    #  tickers_colvec = np.array(s.tickers).reshape(len(s.tickers), 1)
    #  df_data = np.hstack((tickers_colvec, csv_data))
    #  column_headers = ["Input",
    #                    "Positive Tail Exponent",
    #                    "Negative Tail Exponent",
    #                    "Positive Tail xmin",
    #                    "Negative Tail xmin",
    #                    "Positive Tail S.Err",
    #                    "Negative Tail S.Err",
    #                    "Positive Tail Size",
    #                    "Negative Tail Size",
    #                    "Positive Tail KS p-value",
    #                    "Negative Tail KS p-value"]
    #  df = pd.DataFrame(df_data, columns=column_headers)
    #  df.to_csv(filename, index=False)


#  elif approach == "rolling" or approach == "increasing":

#  def run_rolling_increase():
if s.approach == "rolling" or s.approach == "increasing":

    #  question      = "Do you want to save the sequential scaling plot?"
    #  choices      = ['Yes', 'No']
    #  plot_storing = eg.choicebox(question, 'Plot', choices)
    #  plot_storing = "No"
    #
    #  if plot_storing == "Yes":
    #      question = "What is the target directory for the pictures?"
    #      motherpath = eg.enterbox(
    #          question,
    #          title="path",
    #          default = ("C:\Users\\alber\Dropbox\Research"
    #                     "\IP\Econophysics\Final Code Hurst Exponent\\"),
    #      )

    # TODO: add lists below to results_lists_init function?
    #  boxplot_mat = boxplot_mat_init()
    alpha_bpmat = structs.init_alpha_bpmat()

    for t, tck in enumerate(s.ticker_df):

        #  if plot_storing == "Yes":
        #      directory = motherpath + "PowerLawAnimation\\" + labels[i - 1]
        #      try:
        #          os.makedirs(directory)
        #      except OSError:
        #          if not os.path.isdir(directory):
        #              raise
        #      os.chdir(directory)

        # init 2D-array containing results to be written to CSV (per ticker)
        csv_array = structs.init_csv_array()

        #  for l, dt in enumerate(s.spec_dates, start=s.ind_i):
        for i, dt in enumerate(s.spec_dates):

            ll = s.ind_i + i
            lbk = (ll if s.approach == "rolling" else s.ind_i) - s.lookback + 1

            # NOTE: must convert Series to PandasArray to remove Index,
            # otherwise all operations will be aligned on their indexes
            series = s.db_df[tck].iloc[lbk: ll + 1].array
            #  series_0 = s.full_dbdf.iloc[lbk:ll+1, t].array
            #  assert series == series_0

            begin_date = s.db_dates[lbk]
            end_date = dt

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

            print(f"I am analyzing the time series for {tck} "
                  f"between {begin_date} and {end_date}")

            csv_array[i, :] = calc.get_results_tup(series)

            #  # Plot Storing if-block
            #  if plot_storing == "Yes":
            #
            #      if tail_selected == "right" or tail_selected == "both":
            #
            #          plt.figure(
            #              "right tail scaling for "
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
            #              "right-tail scaling_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #          plt.figure("right tail comparison for " + labels[i - 1])
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
            #              "right-tail fitting comparison_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #      if tail_selected == "left" or tail_selected == "both":
            #
            #          plt.figure(
            #              "left tail scaling for "
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
            #              "left-tail scaling_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #          plt.figure("left tail comparison for " + labels[i - 1])
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
            #              "left-tail fitting comparison_"
            #              + begin_date
            #              + "_"
            #              + end_date
            #              + "_"
            #              + labels[i - 1]
            #              + ".jpg"
            #          )
            #          plt.close()

        # Write Tail Statistics to CSV file
        #  FIXME: data_io.write_csv_stats(tail_statistics)

        # getting vectors required for plotting
        plot_vecs_tup = calc.get_plot_vecs(csv_array.T)

        # NOTE: these are used for the boxplots
        # ----> treat w/ care when adding multiprocessing
        #  if s.tail_selected == "right" or s.tail_selected == "both":
        #      boxplot_mat["pos_α_mat"].append(results["pos_α_vec"])
        #  if s.tail_selected == "left" or s.tail_selected == "both":
        #      boxplot_mat["neg_α_mat"].append(results["neg_α_vec"])
        alphas = plot_vecs_tup[0]
        r = len(alphas)  # 1 if only uses 1 tail, 2 if both
        # FIXME: need to group by tails_used, and label to plot
        alpha_bpmat[r*t: r*(t+1)] = alphas
        #  print(alphas)
        #  print(r*t, r*(t+1))
        #  print(alpha_bpmat)

        # add appropriate labels to vectors to be plotted
        plot_vecs = structs.label_plot_vecs(plot_vecs_tup)
        # TODO: move transpose taking into function??

        # TODO: consider doing all plotting at very end of script

        # Plot the alpha exponent in time (right/left/both tail)
        # AND plot the histograms for the rolling alpha
        trp.tabled_figure_plotter(tck, s, plot_vecs)

        # Plot the alpha exponent confidence interval in time
        # and the other 3 time rolling plots
        trp.time_rolling_plotter(tck, s, plot_vecs)
        # FIXME: the above does not plot left tails even with both selected

    #  # Plot the boxplots for the alpha tail(s))
    # TODO: avoid making an array copy, and just use needed vectors directly
    #  pfbx.boxplot(s.tickers, alpha_bpmat, s, show_plot=True)
