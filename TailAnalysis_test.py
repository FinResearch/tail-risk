from utils.settings import settings as s

from utils import ui, structs, calc

import plot_funcs.tail_risk_plotter as trp
#  import plot_funcs.boxplot as pfbx


# TODO: factor plot making & storing code sections
# TODO: use config file (json, yaml, toml) for attr. (color, width, etc.)
# NOTES & IDEAS: create map (json) from plot data to its title, labels, etc.
# NOTE on refactor order: alpha-fit, time-rolling (4 sets), histogram, boxplot
# ASK: plots shown vs. stored are different -> why not store own plots too???


# Execution logic for the actual calculations

#  if approach == "static":
#
#      # TODO: add list below to results_lists_init function?
#      tail_statistics = []
#
#      for i in range(1, N, 1):
#
#          print("I am analyzing the time series for " +
#                tickers[i - 1] + " between " + dates[0] + " and " + dates[-1])
#          series = database[i][initial_index: (final_index + 1)]
#
#          print("You opted for the analysis of the " + return_type)
#          if return_type == "Returns":
#              X = np.array(series[tau:]) - \
#                  np.array(series[0: (len(series) - tau)])
#              lab = "P(t+" + str(tau) + ") - P(t)"
#          elif return_type == "Relative returns":
#              X = np.array(series[tau:]) / \
#                  np.array(series[0: (len(series) - tau)]) - 1.0
#              lab = "P(t+" + str(tau) + ")/P(t) - 1.0"
#          else:
#              X = np.log(
#                  np.array(series[tau:]) /
#                  np.array(series[0: (len(series) - tau)])
#              )
#              lab = r"$\log$" + "(P(t+" + str(tau) + ")/P(t))"
#
#          # TODO: replace below with standardize_tail() but for "Full Series"
#          #  if standardize == "Yes":
#          #      if standardize_target == "Full Series":
#          #          print("I am standardizing your time series")
#          #          S = X
#          #          m = np.mean(S)
#          #          v = np.std(S)
#          #          X = (S - m) / v
#
#          # TODO: replace below with absolutize_tail()
#          #  if abs_value == "Yes":
#          #      if abs_target == "Full Series":
#          #          print("I am taking the absolute value of your time series")
#          #          X = np.abs(X)
#          #          lab = "|" + lab + "|"
#
#          tail_plus, tail_neg, fit_1, fit_2 = fit_tail(tail_selected, X)
#
#          # TODO: when only right or left tail selected,
#          #       the other fit object will be None
#          alpha1 = fit_1.power_law.alpha
#          xmin1 = fit_1.power_law.xmin
#          s_err1 = fit_1.power_law.sigma
#
#          alpha2 = fit_2.power_law.alpha
#          xmin2 = fit_2.power_law.xmin
#          s_err2 = fit_2.power_law.sigma
#
#          if tail_selected == "right" or tail_selected == "both":
#              p1 = plpva.plpva(tail_plus, float(xmin1), "reps", c_iter, "silent")
#              results["pos_α_ks"].append(p1[0])
#
#          if tail_selected == "left" or tail_selected == "both":
#              p2 = plpva.plpva(tail_neg, float(xmin2), "reps", c_iter, "silent")
#              results["neg_α_ks"].append(p2[0])
#
#          # Figures Plot & Show Sections below
#          if tail_selected == "right" or tail_selected == "both":
#
#              plt.figure("right tail scaling for " + tickers[i - 1])
#              z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#              fig4 = fit_1.plot_ccdf(color="b", linewidth=2,
#                                     label="Empirical CCDF")
#              fit_1.power_law.plot_ccdf(
#                  color="b", linestyle="-", label="Fitted CCDF", ax=fig4
#              )
#              fit_1.plot_pdf(color="r", linewidth=2,
#                             label="Empirical PDF", ax=fig4)
#              fit_1.power_law.plot_pdf(
#                  color="r", linestyle="-", label="Fitted PDF", ax=fig4
#              )
#              fig4.set_title(
#                  "Log-log plot of the scaling properties of the right-tail for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              fig4.grid()
#              fig4.legend()
#              col_labels = [r"$\hat{\alpha}$", "Standard err.",
#                            r"$x_{min}$", "size"]
#              table_vals = []
#              table_vals.append(
#                  [
#                      np.round(alpha1, 4),
#                      np.round(s_err1, 4),
#                      np.round(xmin1, 4),
#                      len(filter(lambda x: x > xmin1, tail_plus)),
#                  ]
#              )
#              the_table = plt.table(
#                  cellText=table_vals,
#                  cellLoc="center",
#                  colLabels=col_labels,
#                  loc="bottom",
#                  bbox=[0.0, -0.26, 1.0, 0.10],
#              )
#              the_table.auto_set_font_size(False)
#              the_table.set_fontsize(10)
#              the_table.scale(0.5, 0.5)
#              plt.show()
#
#              plt.figure("right tail comparison for " + tickers[i - 1])
#              fig4 = fit_1.plot_ccdf(color="b", linewidth=2,
#                                     label="Empirical CCDF")
#              fit_1.power_law.plot_ccdf(
#                  color="r", linestyle="-", label="Fitted PL", ax=fig4
#              )
#              fit_1.truncated_power_law.plot_ccdf(
#                  color="g", linestyle="-", label="Fitted TPL", ax=fig4
#              )
#              fit_1.exponential.plot_ccdf(
#                  color="c", linestyle="-", label="Fitted Exp.", ax=fig4
#              )
#              fit_1.lognormal.plot_ccdf(
#                  color="m", linestyle="-", label="Fitted LogN.", ax=fig4
#              )
#              fig4.set_title(
#                  "Comparison of the distributions fitted on the right-tail for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              fig4.grid()
#              fig4.legend()
#              plt.show()
#
#              distribution_list = ["truncated_power_law",
#                                   "exponential", "lognormal"]
#              for pdf in distribution_list:
#                  R, p = fit_1.distribution_compare(
#                      "power_law", pdf, normalized_ratio=True)
#                  #  loglikelihood_ratio_right.append(R)
#                  results["loglr_right"].append(R)
#                  #  loglikelihood_pvalue_right.append(p)
#                  results["loglpv_right"].append(p)
#
#              z.figure("Log Likelihood ratio for the right tail for " +
#                       tickers[i - 1])
#              #  z.bar(
#              #      np.arange(0, len(loglikelihood_ratio_right), 1),
#              #      loglikelihood_ratio_right, 1,)
#              z.bar(
#                  np.arange(0, len(results["loglr_right"]), 1),
#                  results["loglr_right"], 1,)
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
#                       distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              # z.show()
#
#              z.figure("Log Likelihood ratio p-values for the right tail for " +
#                       tickers[i - 1])
#              #  z.bar(
#              #      np.arange(0, len(loglikelihood_pvalue_right), 1),
#              #      loglikelihood_pvalue_right, 1,)
#              z.bar(
#                  np.arange(0, len(results["loglpv_right"]), 1),
#                  results["loglpv_right"], 1,)
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
#                       distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio p values for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              # z.show()
#
#          if tail_selected == "left" or tail_selected == "both":
#
#              plt.figure("left tail scaling for " + tickers[i - 1])
#              z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#              fig4 = fit_2.plot_ccdf(color="b", linewidth=2,
#                                     label="Empirical CCDF")
#              fit_2.power_law.plot_ccdf(
#                  color="b", linestyle="-", label="Fitted CCDF", ax=fig4
#              )
#              fit_2.plot_pdf(color="r", linewidth=2,
#                             label="Empirical PDF", ax=fig4)
#              fit_2.power_law.plot_pdf(
#                  color="r", linestyle="-", label="Fitted PDF", ax=fig4
#              )
#              fig4.set_title(
#                  "Log-log plot of the scaling properties of the left-tail for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              fig4.grid()
#              fig4.legend()
#              col_labels = [r"$\hat{\alpha}$", "Standard err.",
#                            r"$x_{min}$", "size"]
#              table_vals = []
#              table_vals.append(
#                  [
#                      np.round(alpha2, 4),
#                      np.round(s_err2, 4),
#                      np.round(xmin2, 4),
#                      len(filter(lambda x: x > xmin2, tail_neg)),
#                  ]
#              )
#              the_table = plt.table(
#                  cellText=table_vals,
#                  cellLoc="center",
#                  colLabels=col_labels,
#                  loc="bottom",
#                  bbox=[0.0, -0.26, 1.0, 0.10],
#              )
#              the_table.auto_set_font_size(False)
#              the_table.set_fontsize(10)
#              the_table.scale(0.5, 0.5)
#              plt.show()
#
#              plt.figure("left tail comparison for " + tickers[i - 1])
#              fig4 = fit_2.plot_ccdf(color="b", linewidth=2,
#                                     label="Empirical CCDF")
#              fit_2.power_law.plot_ccdf(
#                  color="r", linestyle="-", label="Fitted PL", ax=fig4
#              )
#              fit_2.truncated_power_law.plot_ccdf(
#                  color="g", linestyle="-", label="Fitted TPL", ax=fig4
#              )
#              fit_2.exponential.plot_ccdf(
#                  color="c", linestyle="-", label="Fitted Exp.", ax=fig4
#              )
#              fit_2.lognormal.plot_ccdf(
#                  color="m", linestyle="-", label="Fitted LogN.", ax=fig4
#              )
#              fig4.set_title(
#                  "Comparison of the distributions fitted on the left-tail for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              fig4.grid()
#              fig4.legend()
#              plt.show()
#
#              distribution_list = ["truncated_power_law",
#                                   "exponential", "lognormal"]
#              for pdf in distribution_list:
#                  R, p = fit_2.distribution_compare(
#                      "power_law", pdf, normalized_ratio=True
#                  )
#                  #  loglikelihood_ratio_left.append(R)
#                  results["loglr_left"].append(R)
#                  #  loglikelihood_pvalue_left.append(p)
#                  results["loglpv_left"].append(p)
#
#              z.figure("Log Likelihood ratio for the left tail for " +
#                       tickers[i - 1])
#              #  z.bar(
#              #      np.arange(0, len(loglikelihood_ratio_left), 1),
#              #      loglikelihood_ratio_left, 1,)
#              z.bar(
#                  np.arange(0, len(results["loglr_left"]), 1),
#                  results["loglr_left"], 1,)
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
#                       distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              # z.show()
#
#              z.figure(
#                  "Log Likelihood ratio p-values for the left tail for " +
#                  tickers[i - 1])
#              #  z.bar(
#              #      np.arange(0, len(loglikelihood_pvalue_left), 1),
#              #      loglikelihood_pvalue_left, 1,)
#              z.bar(
#                  np.arange(0, len(results["loglpv_left"]), 1),
#                  results["loglpv_left"], 1,)
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1),
#                       distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio p values for "
#                  + tickers[i - 1]
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              # z.show()
#
#          if tail_selected == "right" or tail_selected == "both":
#
#              results["pos_α_vec"].append(alpha1)
#              results["pos_up_bound"].append(
#                  alpha1 + (st.norm.ppf(1 - multiplier * significance)) * s_err1)
#              results["pos_low_bound"].append(
#                  alpha1 - (st.norm.ppf(1 - multiplier * significance)) * s_err1)
#              results["pos_abs_len"].append(len(filter(lambda x: x >= xmin1,
#                                                       tail_plus)))
#              results["pos_rel_len"].append(
#                  len(filter(lambda x: x >= xmin1, tail_plus)) /
#                  float(len(tail_plus)))
#
#          if tail_selected == "left" or tail_selected == "both":
#              results["neg_α_vec"].append(alpha2)
#              results["neg_up_bound"].append(
#                  alpha2 + (st.norm.ppf(1 - multiplier * significance)) * s_err2)
#              results["neg_low_bound"].append(
#                  alpha2 - (st.norm.ppf(1 - multiplier * significance)) * s_err2)
#              results["neg_abs_len"].append(len(filter(lambda x: x >= xmin2,
#                                                       tail_neg)))
#              results["neg_rel_len"].append(
#                  len(filter(lambda x: x >= xmin2, tail_neg)) /
#                  float(len(tail_neg)))
#
#          # Building tail statistics section
#
#          if tail_selected == "right" or tail_selected == "both":
#              tstat_right = get_tail_stats(fit_1, tail_plus, p1)
#          if tail_selected == "left" or tail_selected == "both":
#              tstat_left = get_tail_stats(fit_2, tail_neg, p2)
#
#          if tail_selected == "both":
#              row = tail_stat_zipper(tstat_right, tstat_left)
#          elif tail_selected == "right":
#              row = tail_stat_zipper(tstat_right, np.zeros(len(tstat_right)))
#          elif tail_selected == "left":
#              row = tail_stat_zipper(np.zeros(len(tstat_left)), tstat_left)
#
#          tail_statistics.append(row)
#
#      # Preparing the figure
#
#      z.figure("static alpha")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      if tail_selected == "right" or tail_selected == "both":
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  positive_alpha_vec,
#              results["pos_α_vec"],
#              marker="^",
#              markersize=10.0,
#              linewidth=0.0,
#              color="green",
#              label="right tail",
#          )
#      if tail_selected == "left" or tail_selected == "both":
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  negative_alpha_vec,
#              results["neg_α_vec"],
#              marker="^",
#              markersize=10.0,
#              linewidth=0.0,
#              color="red",
#              label="left tail",
#          )
#      z.xticks(range(1, len(tickers) + 1, 1), tickers)
#      z.xlim(xmin=0.5, xmax=len(tickers) + 0.5)
#      z.ylabel(r"$\alpha$")
#      z.title(
#          "Estimation of the "
#          + r"$\alpha$"
#          + "-right tail exponents using KS-Method"
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input series: "
#          + lab
#      )
#      z.legend(
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=2,
#          mode="expand", borderaxespad=0
#      )
#      z.grid()
#      # z.show()
#
#      if tail_selected == "right" or tail_selected == "both":
#
#          # Confidence interval for the right tail
#          z.figure("Confidence interval for the right tail")
#          z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  positive_alpha_vec,
#              results["pos_α_vec"],
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="green",
#              label="right tail",
#          )
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  positive_upper_bound,
#              results["pos_up_bound"],
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="purple",
#              label="Upper bound",
#          )
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  positive_lower_bound,
#              results["pos_low_bound"],
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="blue",
#              label="Lower bound",
#          )
#          z.plot(
#              range(0, len(tickers) + 2, 1), np.repeat(3, len(tickers) + 2),
#              color="red"
#          )
#          z.plot(
#              range(0, len(tickers) + 2, 1), np.repeat(2, len(tickers) + 2),
#              color="red"
#          )
#          z.xticks(range(1, len(tickers) + 1, 1), tickers)
#          z.xlim(xmin=0.5, xmax=len(tickers) + 0.5)
#          z.ylabel(r"$\alpha$")
#          z.title(
#              "Confidence intervals for the "
#              + r"$\alpha$"
#              + "-right tail exponents "
#              + "(c = "
#              + str(1 - significance)
#              + ")"
#              + "\n"
#              + "Time Period: "
#              + dates[0]
#              + " - "
#              + dates[-1]
#              + ". Input series: "
#              + lab
#          )
#          z.legend(
#              bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
#              ncol=3,
#              mode="expand",
#              borderaxespad=0,
#          )
#          z.grid()
#          # z.show()
#
#      if tail_selected == "left" or tail_selected == "both":
#
#          # Confidence interval for the left tail
#          z.figure("Confidence interval for the left tail")
#          z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  negative_alpha_vec,
#              results["neg_α_vec"],
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="green",
#              label="left tail",
#          )
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  negative_upper_bound,
#              results["neg_up_bound"],
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="purple",
#              label="Upper bound",
#          )
#          z.plot(
#              range(1, len(tickers) + 1, 1),
#              #  negative_lower_bound,
#              results["neg_low_bound"],
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="blue",
#              label="Lower bound",
#          )
#          z.plot(
#              range(0, len(tickers) + 2, 1), np.repeat(3, len(tickers) + 2),
#              color="red"
#          )
#          z.plot(
#              range(0, len(tickers) + 2, 1), np.repeat(2, len(tickers) + 2),
#              color="red"
#          )
#          z.xticks(range(1, len(tickers) + 1, 1), tickers)
#          z.xlim(xmin=0.5, xmax=len(tickers) + 0.5)
#          z.ylabel(r"$\alpha$")
#          z.title(
#              "Confidence intervals for the "
#              + r"$\alpha$"
#              + "-left tail exponents "
#              + "(c = "
#              + str(1 - significance)
#              + ")"
#              + "\n"
#              + "Time Period: "
#              + dates[0]
#              + " - "
#              + dates[-1]
#              + ". Input series: "
#              + lab
#          )
#          z.legend(
#              bbox_to_anchor=(0.0, -0.175, 1.0, 0.02),
#              ncol=3,
#              mode="expand",
#              borderaxespad=0,
#          )
#          z.grid()
#          # z.show()
#
#      # Absolute length of the tail bar chart
#
#      z.figure("Absolute tail lengths")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      amplitude = 0.5
#      if tail_selected == "right" or tail_selected == "both":
#          z.bar(
#              np.arange(0, 2 * len(tickers), 2),
#              #  positive_abs_length,
#              results["pos_abs_len"],
#              amplitude,
#              color="green",
#              label="right tail",
#          )
#      if tail_selected == "left" or tail_selected == "both":
#          z.bar(
#              np.arange(amplitude, 2 * len(tickers) + amplitude, 2),
#              #  negative_abs_length,
#              results["neg_abs_len"],
#              amplitude,
#              color="red",
#              label="left tail",
#          )
#      z.xticks(np.arange(amplitude, 2 * len(tickers) + amplitude, 2), tickers)
#      z.ylabel("Tail length")
#      z.title(
#          "Bar chart representation of the length of the tails"
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input series: "
#          + lab
#      )
#      z.legend(
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3, mode="expand",
#          borderaxespad=0
#      )
#      z.grid()
#      # z.show()
#
#      # Absolute length of the tail bar chart
#
#      z.figure("Relative tail lengths")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      amplitude = 0.5
#      if tail_selected == "right" or tail_selected == "both":
#          z.bar(
#              np.arange(0, 2 * len(tickers), 2),
#              #  positive_rel_length,
#              results["pos_rel_len"],
#              amplitude,
#              color="green",
#              label="right tail",
#          )
#      if tail_selected == "left" or tail_selected == "both":
#          z.bar(
#              np.arange(amplitude, 2 * len(tickers) + amplitude, 2),
#              #  negative_rel_length,
#              results["neg_rel_len"],
#              amplitude,
#              color="red",
#              label="left tail",
#          )
#      z.xticks(np.arange(amplitude, 2 * len(tickers) + amplitude, 2), tickers)
#      z.ylabel("Tail relative length")
#      z.title(
#          "Bar chart representation of the relative length of the tails"
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input series: "
#          + lab
#      )
#      z.legend(
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3,
#          mode="expand", borderaxespad=0
#      )
#      z.grid()
#      # z.show()
#
#      # KS test outcome
#
#      z.figure("KS test p value for the tails")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      amplitude = 0.5
#      if tail_selected == "right" or tail_selected == "both":
#          z.bar(
#              np.arange(0, 2 * len(tickers), 2),
#              #  positive_alpha_KS,
#              results["pos_α_ks"],
#              amplitude,
#              color="green",
#              label="right tail",
#          )
#      if tail_selected == "left" or tail_selected == "both":
#          z.bar(
#              np.arange(amplitude, 2 * len(tickers) + amplitude, 2),
#              #  negative_alpha_KS,
#              results["neg_α_ks"],
#              amplitude,
#              color="red",
#              label="left tail",
#          )
#      z.xticks(np.arange(amplitude, 2 * len(tickers) + amplitude, 2), tickers)
#      z.ylabel("p-value")
#      z.title(
#          "KS-statistics: p-value obtained from Clauset algorithm"
#          + "\n"
#          + "Time Period: "
#          + dates[0]
#          + " - "
#          + dates[-1]
#          + ". Input series: "
#          + lab
#      )
#      z.legend(
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3,
#          mode="expand", borderaxespad=0
#      )
#      z.grid()
#      # z.show()
#
#      # Write Tail Statistics to CSV file
#      filename = "TailStatistics_Overall.csv"
#      tickers_colvec = np.array(tickers).reshape(len(tickers), 1)
#      df_data = np.hstack((tickers_colvec, tail_statistics))
#      column_headers = ["Input",
#                        "Positive Tail Exponent",
#                        "Negative Tail Exponent",
#                        "Positive Tail xmin",
#                        "Negative Tail xmin",
#                        "Positive Tail S.Err",
#                        "Negative Tail S.Err",
#                        "Positive Tail Size",
#                        "Negative Tail Size",
#                        "Positive Tail KS p-value",
#                        "Negative Tail KS p-value"]
#      df = pd.DataFrame(df_data, columns=column_headers)
#      df.to_csv(filename, index=False)

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

    for tck in s.ticker_df:

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
            # ASK: is the none "rolling" approach, "increasing"?
            lbk = (ll if s.approach == "rolling" else s.ind_i) - s.lookback + 1

            # NOTE: must convert Series to PandasArray to remove Index,
            # otherwise all operations will be aligned on their indexes
            series = s.db_df[tck].iloc[lbk: ll + 1].array

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
        plot_vecs = structs.label_plot_vecs(calc.get_plot_vecs(csv_array.T))
        # TODO: move transpose taking into function??

        # NOTE: these are used for the boxplots
        # ----> treat w/ care when adding multiprocessing
        #  if s.tail_selected == "right" or s.tail_selected == "both":
        #      boxplot_mat["pos_α_mat"].append(results["pos_α_vec"])
        #  if s.tail_selected == "left" or s.tail_selected == "both":
        #      boxplot_mat["neg_α_mat"].append(results["neg_α_vec"])

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
    #  pfbx.boxplot(tickers, boxplot_mat, settings, show_plot=True)
