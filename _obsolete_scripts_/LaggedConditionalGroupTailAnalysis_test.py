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


# import own modules

from settings import settings as s
import utils
import plpva as plpva

#  eng = matlab.engine.start_matlab()
#  directory = os.getcwd()
#  eng.cd(directory, nargout=0)

#####################################
# Tools Functions                   #
#####################################


#  def Extractor(filename, tickers):
#      object = pd.read_csv(filename)
#      output = [(object["Date"].values).tolist()]
#      for i in range(0, len(tickers), 1):
#          try:
#              output.append((object[tickers[i]].values).tolist())
#          except KeyError:
#              print("Ticker " + tickers[i] + " not found in " + filename)
#      return output


#####################################
# Script begins                     #
#####################################

#  #  question       = "What is the name of the database?"
#  #  database_name = eg.enterbox(question, title="DB name", default="dbMSTR_test.csv")
#  database_name = "dbMSTR_test.csv"
#  #  database_name = "dbMarkitTest.xlsx"


#  question       = "How many entries would you like to analyze?"
#  no_entries       = int(eg.enterbox(question, title="No. entries", default="5"))
no_entries = 4
fieldNames = ["# " + str(i) for i in range(1, no_entries + 1, 1)]
fieldValues = [
    "DE 01Y",
    "DE 03Y",
    "DE 05Y",
    "DE 10Y",  #'DE 30Y',
    #  'FR 01Y', 'FR 03Y', 'FR 05Y', 'FR 10Y', 'FR 30Y',
    #  'ES 01Y', 'ES 03Y', 'ES 05Y', 'ES 10Y', 'ES 30Y',
    #  'PT 01Y', 'PT 03Y', 'PT 05Y', 'PT 10Y', 'PT 30Y',
    #  'IT 01Y', 'IT 03Y', 'IT 05Y', 'IT 10Y', 'IT 30Y',
    #  'IR 01Y', 'IR 03Y', 'IR 05Y', 'IR 10Y', 'IR 30Y'
]

#  fieldValues   = [
#                   'DE 01Y', 'DE 02Y', 'DE 03Y', 'DE 04Y', 'DE 05Y', 'DE 07Y', 'DE 10Y', 'DE 15Y', 'DE 20Y', 'DE 30Y',
#                   'AU 01Y', 'AU 02Y', 'AU 03Y', 'AU 04Y', 'AU 05Y', 'AU 07Y', 'AU 10Y', 'AU 15Y', 'AU 20Y', 'AU 30Y',
#                   'CY 01Y', 'CY 02Y', 'CY 03Y', 'CY 04Y', 'CY 05Y', 'CY 07Y', 'CY 10Y', 'CY 15Y', 'CY 20Y', 'CY 30Y',
#                   'EE 01Y', 'EE 02Y', 'EE 03Y', 'EE 04Y', 'EE 05Y', 'EE 07Y', 'EE 10Y', 'EE 15Y', 'EE 20Y', 'EE 30Y',
#                   'ES 01Y', 'ES 02Y', 'ES 03Y', 'ES 04Y', 'ES 05Y', 'ES 07Y', 'ES 10Y', 'ES 15Y', 'ES 20Y', 'ES 30Y',
#                   'FI 01Y', 'FI 02Y', 'FI 03Y', 'FI 04Y', 'FI 05Y', 'FI 07Y', 'FI 10Y', 'FI 15Y', 'FI 20Y', 'FI 30Y',
#                   'FR 01Y', 'FR 02Y', 'FR 03Y', 'FR 04Y', 'FR 05Y', 'FR 07Y', 'FR 10Y', 'FR 15Y', 'FR 20Y', 'FR 30Y',
#                   'DE 01Y', 'DE 02Y', 'DE 03Y', 'DE 04Y', 'DE 05Y', 'DE 07Y', 'DE 10Y', 'DE 15Y', 'DE 20Y', 'DE 30Y',
#                   'IR 01Y', 'IR 02Y', 'IR 03Y', 'IR 04Y', 'IR 05Y', 'IR 07Y', 'IR 10Y', 'IR 15Y', 'IR 20Y', 'IR 30Y',
#                   'IT 01Y', 'IT 02Y', 'IT 03Y', 'IT 04Y', 'IT 05Y', 'IT 07Y', 'IT 10Y', 'IT 15Y', 'IT 20Y', 'IT 30Y',
#                   'LV 01Y', 'LV 02Y', 'LV 03Y', 'LV 04Y', 'LV 05Y', 'LV 07Y', 'LV 10Y', 'LV 15Y', 'LV 20Y', 'LV 30Y',
#                   'LT 01Y', 'LT 02Y', 'LT 03Y', 'LT 04Y', 'LT 05Y', 'LT 07Y', 'LT 10Y', 'LT 15Y', 'LT 20Y', 'LT 30Y',
#                   'NL 01Y', 'NL 02Y', 'NL 03Y', 'NL 04Y', 'NL 05Y', 'NL 07Y', 'NL 10Y', 'NL 15Y', 'NL 20Y', 'NL 30Y',
#                   'PT 01Y', 'PT 02Y', 'PT 03Y', 'PT 04Y', 'PT 05Y', 'PT 07Y', 'PT 10Y', 'PT 15Y', 'PT 20Y', 'PT 30Y',
#                   'SK 01Y', 'SK 02Y', 'SK 03Y', 'SK 04Y', 'SK 05Y', 'SK 07Y', 'SK 10Y', 'SK 15Y', 'SK 20Y', 'SK 30Y',
#                   'SI 01Y', 'SI 02Y', 'SI 03Y', 'SI 04Y', 'SI 05Y', 'SI 07Y', 'SI 10Y', 'SI 15Y', 'SI 20Y', 'SI 30Y',
#  ]

#  title           = "Tickers input"
#  labels        = eg.multenterbox(question, title, fieldNames, fieldValues)
labels = fieldValues

#  database = Extractor(database_name, labels)
#  #  question      = "Please specify the initial date, the final date and the lookback for rolling analysis?"
#  #  fieldNames    = ["Initial Date", "Final Date", "Lookback"]
#  #  fieldValues   = ["17-07-03", "5/5/2016", "252"]
#  #  title         = "Dates input"
#  #  input         = eg.multenterbox(question, title, fieldNames, fieldValues)
#  #  initial_date  = input[0]
#  #  final_date    = input[1]
#  #  lookback      = int(input[2])
#  #  initial_date = "1/1/2016"
#  initial_date = "31-03-16"
#  final_date = "5/5/2016"
#  lookback = 252
#
#
#  #  question      = "Please specify which type of series you want to study"
#  #  choices       = ['Returns', 'Relative Returns','Log-Returns']
#  #  input_type    = eg.choicebox(question, 'Input type', choices)
#  input_type = "Log-Returns"
#
#  #  msg           = "Please specify the time lag of the input series: 1 = daily, 5 = weekly, 22 = monthly"
#  #  tau           = int(eg.enterbox(msg, title="delta", default="1"))
#  tau = 1
#
#  #  question      = "Do you want to normalize each investigated time series?"
#  #  choices       = ['Yes', 'No']
#  #  standardize   = eg.buttonbox(question, 'Normalization', choices)
#  standardize = "No"
#
#  #  if standardize == "Yes":
#  #      question = "When do you want to standardize your series with respect to the grouping procedure?"
#  #      choices = ["Before", "After", "Both"]
#  #      title = "Normalization timing"
#  #      norm_timing = eg.choicebox(question, title, choices)
#
#  #  question      = "Do you want to take the absolute value of your series (after any eventual normalization)?"
#  #  choices       = ['Yes', 'No']
#  #  abs_value     = eg.buttonbox(question, 'Absolute value', choices)
#  abs_value = "No"
#
#  #  if abs_value == "Yes":
#  #      question = "When do you want to take the absolute value of your series with respect to the grouping procedure?"
#  #      choices = ["Before", "After", "Both"]
#  #      title = "Absolute value timing"
#  #      abs_timing = eg.choicebox(question, title, choices)
#
#  #  question      = "Please specify which approach you would like to use"
#  #  choices       = ['Static', 'Rolling', 'Increasing']
#  #  approach      = eg.choicebox(question, 'Approach', choices)
#  approach = "Rolling"
#  #  an_freq = 1

#  question       = "How do you want to group the inputs?"
#  choices       = ['Country', 'Maturity', 'Core vs Peripheral', 'All', 'Rating', 'High Yield', 'Investment Grade']
#  title         = 'Partition type'
#  partition     = eg.choicebox(question, title, choices)
partition = "Country"

if partition == "Country":
    identifiers = []
    for lab in labels:
        country = lab[0:2]
        identifiers.append(country)
#  elif partition == "Maturity":
#      identifiers = []
#      for lab in labels:
#          maturity = lab[3:6]
#          identifiers.append(maturity)
#  elif partition == "Core vs Peripheral":
#      clusters = [["DE", "FR", "BE"], ["IT", "ES", "PT", "IR", "GR"]]
#      identifiers = []
#      for lab in labels:
#          country = lab[0:2]
#          if country in clusters[0]:
#              identifiers.append("Core Countries")
#          elif country in clusters[1]:
#              identifiers.append("Peripheral Countries")
#          else:
#              print(
#                  "The current country ("
#                  + country
#                  + ") is not listed among core or peripheral groups"
#              )
#              exit
#  elif partition == "All":
#      identifiers = []
#      for lab in labels:
#          identifiers.append("All Countries")
#  else:
#      "Nada"
print(f"You have decided to group your data by {partition}")

#  if approach != 'Static':
#  question        = "Please specify the amplitude of the sliding window (days) in the non static analysis"
#  sliding_window  = int(eg.enterbox(question, title="block length", default="1"))
sliding_window = 1
# ASK/NOTE: the above same as an_freq in TailAnalysis??


#  #  question      = "Please specify which tail you want to plot in the alpha timeline"
#  #  choices       = ['Left', 'Right', 'Both']
#  #  tail_selected = eg.choicebox(question, 'Select a tail', choices)
#  tail_selected = "Both"
#
#  #  question      = "What is the nature of your data?"
#  #  choices       = ['Discrete', 'Continuous']
#  #  data_nature   = eg.choicebox(question, 'Data type', choices)
#  data_nature = "Continuous"

#  question      = "What is the criteria for picking xmin"
#  choices       = ['Clauset', 'Rolling', 'Manual', 'Percentile']
#  xmin_rule     = eg.choicebox(question, 'xmin rule', choices)
xmin_rule = "Rolling"

#  if xmin_rule == 'Rolling':
#          question     = "How many days do you want to include in the moving xmin average?"
#          rolling_days = int(eg.enterbox(question, title="rolling days", default="66"))
#          question     = "How many lags do you want in your average?"
#          rolling_lags = int(eg.enterbox(question, title="rolling days", default="0"))
rolling_days = 66
#  rolling_days = 22
rolling_lags = 0

if xmin_rule == "Manual":
    question = "What is the value for xmin?"
    xmin_value = float(eg.enterbox(question, title="xmin value", default="0.0"))
if xmin_rule == "Percentile":
    question = "What is the value of the significance for xmin?"
    xmin_sign = float(eg.enterbox(question, title="xmin percentile", default="91.41"))

#  if tail_selected == "Both":
#      multiplier = 0.5
#  else:
#      multiplier = 1.0
#
#  #  msg           = "Please specify the significance of the confidence interval (1 - a) for the alpha parameter "
#  #  significance  = float(eg.enterbox(msg, title="alpha", default="0.05"))
#  significance = 0.05
#
#  #  question      = "What is the number of iterations for the Clauset p-value algorithm?"
#  #  c_iter        = int(eg.enterbox(question, title="iterations", default='2'))
#  c_iter = 2


# NOTE: these lists appear to only be used for plotting
def xmin_lists_init():
    labels = ("xmin_vec_right", "xmin_vec_left",)
    # NOTE: length of each list is the number of days -> so use np.ndarray
    return {l: [] for l in labels}


def results_dicts_init():
    labels = ("pos_α_vec", "neg_α_vec", "pos_α_ks", "neg_α_ks",
              "pos_up_bound", "neg_up_bound", "pos_low_bound", "neg_low_bound",
              "pos_abs_len", "neg_abs_len", "pos_rel_len", "neg_rel_len",
              "loglr_right", "loglr_left", "loglpv_right", "loglpv_left")
    # NOTE: length of each list is the number of days -> so use np.ndarray
    return {d: {} for d in labels}

# lists to store the results for plotting (16 total)
#  results = results_lists_init()
# TODO: zero "results" container on each ticker iteration OR store them all


# Execution logic for the actual calculations

#  if approach == "Static":
#
#      #  positive_alpha_vec = []
#      #  negative_alpha_vec = []
#      #  positive_alpha_KS = []
#      #  negative_alpha_KS = []
#      #  positive_upper_bound = []
#      #  positive_lower_bound = []
#      #  negative_upper_bound = []
#      #  negative_lower_bound = []
#      #  positive_abs_length = []
#      #  positive_rel_length = []
#      #  negative_abs_length = []
#      #  negative_rel_length = []
#      #  x_min_right_vector = []
#      #  x_min_left_vector = []
#
#      initial_index = database[0].index(initial_date)
#      final_index = database[0].index(final_date)
#      dates = database[0][initial_index : (final_index + 1)]
#      labelstep = (
#          22
#          if len(dates) <= 252
#          else 66
#          if (len(dates) > 252 and len(dates) <= 756)
#          else 121
#      )
#      N = len(database)
#
#      tail_statistics = []
#      BlockDict = {}
#
#      if input_type == "Returns":
#          lab = "P(t+" + str(tau) + ") - P(t)"
#      elif input_type == "Relative returns":
#          lab = "P(t+" + str(tau) + ")/P(t) - 1.0"
#      else:
#          lab = r"$\log$" + "(P(t+" + str(tau) + ")/P(t))"
#
#      if abs_value == "Yes":
#          lab = "|" + lab + "|"
#
#      for i in range(1, N, 1):
#
#          loglikelihood_ratio_right = []
#          loglikelihood_pvalue_right = []
#          loglikelihood_ratio_left = []
#          loglikelihood_pvalue_left = []
#
#          print(
#              "I am analyzing the time series for "
#              + labels[i - 1]
#              + " between "
#              + dates[0]
#              + " and "
#              + dates[-1]
#          )
#          series = database[i][initial_index : (final_index + 1)]
#
#          print("You opted for the analysis of the " + input_type)
#          if input_type == "Returns":
#              X = np.array(series[tau:]) - np.array(series[0 : (len(series) - tau)])
#          elif input_type == "Relative returns":
#              X = np.array(series[tau:]) / np.array(series[0 : (len(series) - tau)]) - 1.0
#          else:
#              X = np.log(
#                  np.array(series[tau:]) / np.array(series[0 : (len(series) - tau)])
#              )
#
#          if standardize == "Yes":
#              if norm_timing == "Before" or norm_timing == "Both":
#                  print("I am standardizing your time series")
#                  S = X
#                  m = np.mean(S)
#                  v = np.std(S)
#                  X = (S - m) / v
#
#          if abs_value == "Yes":
#              if abs_timing == "Before" or abs_timing == "Both":
#                  print("I am taking the absolute value of your time series")
#                  X = np.abs(X)
#
#          if identifiers[i - 1] in BlockDict:
#              print(
#                  "I found an existing group under the identifier : "
#                  + identifiers[i - 1]
#                  + ". Your time series is added in that pool"
#              )
#              BlockDict[identifiers[i - 1]].extend(X.tolist())
#          else:
#              print(
#                  "I have not found an existing group under the identifier : "
#                  + identifiers[i - 1]
#                  + ". I create the group and i add your time series is added in that pool"
#              )
#              BlockDict[identifiers[i - 1]] = X.tolist()
#
#      key = BlockDict.keys()
#      for el in key:
#
#          print(
#              "I am analyzing the group identified by "
#              + el
#              + " between "
#              + dates[0]
#              + " and "
#              + dates[-1]
#          )
#          X = BlockDict[el]
#
#          if standardize == "Yes":
#              if norm_timing == "After" or norm_timing == "Both":
#                  print("I am standardizing your time series")
#                  S = X
#                  m = np.mean(S)
#                  v = np.std(S)
#                  X = (S - m) / v
#
#          if abs_value == "Yes":
#              if abs_timing == "After" or abs_timing == "Both":
#                  print("I am taking the absolute value of your time series")
#                  X = np.abs(X)
#
#          if tail_selected == "Right" or tail_selected == "Both":
#              tail_plus = X
#
#          if tail_selected == "Left" or tail_selected == "Both":
#              tail_neg = (np.dot(-1.0, tail_plus)).tolist()
#
#          if data_nature == "Continuous":
#              if xmin_rule == "Clauset":
#                  if tail_selected == "Right" or tail_selected == "Both":
#                      fit_1 = pl.Fit(list(filter(lambda x: x != 0, tail_plus)))
#                  if tail_selected == "Left" or tail_selected == "Both":
#                      fit_2 = pl.Fit(list(filter(lambda x: x != 0, tail_neg)))
#              elif xmin_rule == "Manual":
#                  if tail_selected == "Right" or tail_selected == "Both":
#                      fit_1 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_plus), xmin=xmin_value)
#                      )
#                  if tail_selected == "Left" or tail_selected == "Both":
#                      fit_2 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_neg), xmin=xmin_value)
#                      )
#              else:
#                  if tail_selected == "Right" or tail_selected == "Both":
#                      fit_1 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_plus)),
#                          xmin=np.percentile(tail_plus, xmin_sign),
#                      )
#                  if tail_selected == "Left" or tail_selected == "Both":
#                      fit_2 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_neg)),
#                          xmin=np.percentile(tail_neg, xmin_sign),
#                      )
#          else:
#              if xmin_rule == "Clauset":
#                  if tail_selected == "Right" or tail_selected == "Both":
#                      fit_1 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_plus), discrete=True)
#                      )
#                  if tail_selected == "Left" or tail_selected == "Both":
#                      fit_2 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_neg), discrete=True)
#                      )
#              elif xmin_rule == "Manual":
#                  if tail_selected == "Right" or tail_selected == "Both":
#                      fit_1 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_plus)),
#                          discrete=True,
#                          xmin=xmin_value,
#                      )
#                  if tail_selected == "Left" or tail_selected == "Both":
#                      fit_2 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_neg)),
#                          discrete=True,
#                          xmin=xmin_value,
#                      )
#              else:
#                  if tail_selected == "Right" or tail_selected == "Both":
#                      fit_1 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_plus)),
#                          discrete=True,
#                          xmin=np.percentile(tail_plus, xmin_sign),
#                      )
#                  if tail_selected == "Left" or tail_selected == "Both":
#                      fit_2 = pl.Fit(
#                          list(filter(lambda x: x != 0, tail_neg)),
#                          discrete=True,
#                          xmin=np.percentile(tail_neg, xmin_sign),
#                      )
#
#          if tail_selected == "Right" or tail_selected == "Both":
#              alpha1 = fit_1.power_law.alpha
#              xmin1 = fit_1.power_law.xmin
#              s_err1 = fit_1.power_law.sigma
#              p1 = eng.plpva(
#                  matlab.double(np.array(tail_plus).tolist()),
#                  float(xmin1),
#                  "reps",
#                  float(c_iter),
#                  "silent",
#                  nargout=2,
#              )
#              positive_alpha_KS.append(p1[0])
#              x_min_right_vector.append(xmin1)
#
#          if tail_selected == "Left" or tail_selected == "Both":
#              alpha2 = fit_2.power_law.alpha
#              xmin2 = fit_2.power_law.xmin
#              s_err2 = fit_2.power_law.sigma
#              p2 = eng.plpva(
#                  matlab.double(np.array(tail_neg).tolist()),
#                  float(xmin2),
#                  "reps",
#                  float(c_iter),
#                  "silent",
#                  nargout=2,
#              )
#              negative_alpha_KS.append(p2[0])
#              x_min_left_vector.append(xmin2)
#
#          # R, p = fit_1.distribution_compare('power_law', 'exponential',normalized_ratio = True)
#          # print(R)
#          if tail_selected == "Right" or tail_selected == "Both":
#
#              plt.figure("Right tail scaling for group " + el)
#              z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#              fig4 = fit_1.plot_ccdf(color="b", linewidth=2, label="Empirical CCDF")
#              fit_1.power_law.plot_ccdf(
#                  color="b", linestyle="-", label="Fitted CCDF", ax=fig4
#              )
#              fit_1.plot_pdf(color="r", linewidth=2, label="Empirical PDF", ax=fig4)
#              fit_1.power_law.plot_pdf(
#                  color="r", linestyle="-", label="Fitted PDF", ax=fig4
#              )
#              fig4.set_title(
#                  "Log-log plot of the scaling properties of the right-tail for the group "
#                  + el
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
#              col_labels = [r"$\hat{\alpha}$", "Standard err.", r"$x_{min}$", "size"]
#              table_vals = []
#              table_vals.append(
#                  [
#                      np.round(alpha1, 4),
#                      np.round(s_err1, 4),
#                      np.round(xmin1, 4),
#                      len(list(filter(lambda x: x > xmin1, tail_plus)),),
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
#              plt.figure("Right tail comparison for group " + el)
#              fig4 = fit_1.plot_ccdf(color="b", linewidth=2, label="Empirical CCDF")
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
#                  "Comparison of the distributions fitted on the right-tail for the group "
#                  + el
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
#              distribution_list = ["truncated_power_law", "exponential", "lognormal"]
#              for pdf in distribution_list:
#                  R, p = fit_1.distribution_compare(
#                      "power_law", pdf, normalized_ratio=True
#                  )
#                  loglikelihood_ratio_right.append(R)
#                  loglikelihood_pvalue_right.append(p)
#
#              z.figure("Log Likelihood ratio for the right tail for group " + el)
#              z.bar(
#                  np.arange(0, len(loglikelihood_ratio_right), 1),
#                  loglikelihood_ratio_right,
#                  1,
#              )
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1), distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio for group "
#                  + el
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              z.show()
#
#              z.figure("Log Likelihood ratio p-values for the right tail for group " + el)
#              z.bar(
#                  np.arange(0, len(loglikelihood_pvalue_right), 1),
#                  loglikelihood_pvalue_right,
#                  1,
#              )
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1), distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio p values for group "
#                  + el
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              z.show()
#
#          if tail_selected == "Left" or tail_selected == "Both":
#
#              plt.figure("Left tail scaling for group " + el)
#              z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#              fig4 = fit_2.plot_ccdf(color="b", linewidth=2, label="Empirical CCDF")
#              fit_2.power_law.plot_ccdf(
#                  color="b", linestyle="-", label="Fitted CCDF", ax=fig4
#              )
#              fit_2.plot_pdf(color="r", linewidth=2, label="Empirical PDF", ax=fig4)
#              fit_2.power_law.plot_pdf(
#                  color="r", linestyle="-", label="Fitted PDF", ax=fig4
#              )
#              fig4.set_title(
#                  "Log-log plot of the scaling properties of the left-tail for group "
#                  + el
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
#              col_labels = [r"$\hat{\alpha}$", "Standard err.", r"$x_{min}$", "size"]
#              table_vals = []
#              table_vals.append(
#                  [
#                      np.round(alpha2, 4),
#                      np.round(s_err2, 4),
#                      np.round(xmin2, 4),
#                      len(list(filter(lambda x: x > xmin2, tail_neg)),),
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
#              plt.figure("Left tail comparison for group " + el)
#              fig4 = fit_2.plot_ccdf(color="b", linewidth=2, label="Empirical CCDF")
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
#                  "Comparison of the distributions fitted on the left-tail for the group "
#                  + el
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
#              distribution_list = ["truncated_power_law", "exponential", "lognormal"]
#              for pdf in distribution_list:
#                  R, p = fit_2.distribution_compare(
#                      "power_law", pdf, normalized_ratio=True
#                  )
#                  loglikelihood_ratio_left.append(R)
#                  loglikelihood_pvalue_left.append(p)
#
#              z.figure("Log Likelihood ratio for the left tail for group " + el)
#              z.bar(
#                  np.arange(0, len(loglikelihood_ratio_left), 1),
#                  loglikelihood_ratio_left,
#                  1,
#              )
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1), distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio for group "
#                  + el
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              z.show()
#
#              z.figure("Log Likelihood ratio p-values for the left tail for group " + el)
#              z.bar(
#                  np.arange(0, len(loglikelihood_pvalue_left), 1),
#                  loglikelihood_pvalue_left,
#                  1,
#              )
#              z.xticks(np.arange(0.5, len(distribution_list) + 0.5, 1), distribution_list)
#              z.ylabel("R")
#              z.title(
#                  "Log-likelihood ratio p values for group "
#                  + el
#                  + "\n"
#                  + "Time Period: "
#                  + dates[0]
#                  + " - "
#                  + dates[-1]
#                  + ". Input series: "
#                  + lab
#              )
#              z.grid()
#              z.show()
#
#          if tail_selected == "Right" or tail_selected == "Both":
#
#              positive_alpha_vec.append(alpha1)
#              positive_upper_bound.append(
#                  alpha1 + (st.norm.ppf(1 - multiplier * significance)) * s_err1
#              )
#              positive_lower_bound.append(
#                  alpha1 - (st.norm.ppf(1 - multiplier * significance)) * s_err1
#              )
#              positive_abs_length.append(
#                  len(list(filter(lambda x: x >= xmin1, tail_plus)))
#              )
#              positive_rel_length.append(
#                  len(
#                      list(filter(lambda x: x >= xmin1, tail_plus))
#                      / float(len(tail_plus))
#                  )
#              )
#
#          if tail_selected == "Left" or tail_selected == "Both":
#              negative_alpha_vec.append(alpha2)
#              negative_upper_bound.append(
#                  alpha2 + (st.norm.ppf(1 - multiplier * significance)) * s_err2
#              )
#              negative_lower_bound.append(
#                  alpha2 - (st.norm.ppf(1 - multiplier * significance)) * s_err2
#              )
#              negative_abs_length.append(
#                  len(list(filter(lambda x: x >= xmin2, tail_neg)))
#              )
#              negative_rel_length.append(
#                  len(list(filter(lambda x: x >= xmin2, tail_neg)) / float(len(tail_neg)))
#              )
#
#          if tail_selected == "Both":
#              row = [
#                  alpha1,
#                  alpha2,
#                  xmin1,
#                  xmin2,
#                  s_err1,
#                  s_err2,
#                  len(list(filter(lambda x: x >= xmin1, tail_plus))),
#                  len(list(filter(lambda x: x >= xmin2, tail_neg))),
#                  p1[0],
#                  p2[0],
#              ]
#          if tail_selected == "Right":
#              row = [
#                  alpha1,
#                  0,
#                  xmin1,
#                  0,
#                  s_err1,
#                  0,
#                  len(list(filter(lambda x: x >= xmin1, tail_plus))),
#                  0,
#                  p1[0],
#                  0,
#              ]
#          if tail_selected == "Left":
#              row = [
#                  0,
#                  alpha2,
#                  0,
#                  xmin2,
#                  0,
#                  s_err2,
#                  0,
#                  len(list(filter(lambda x: x >= xmin2, tail_neg))),
#                  0,
#                  p2[0],
#              ]
#
#          tail_statistics.append(row)
#
#      # Preparing the figure
#
#      z.figure("Static alpha")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      if tail_selected == "Right" or tail_selected == "Both":
#          z.plot(
#              range(1, len(key) + 1, 1),
#              positive_alpha_vec,
#              marker="^",
#              markersize=10.0,
#              linewidth=0.0,
#              color="green",
#              label="Right tail",
#          )
#      if tail_selected == "Left" or tail_selected == "Both":
#          z.plot(
#              range(1, len(key) + 1, 1),
#              negative_alpha_vec,
#              marker="^",
#              markersize=10.0,
#              linewidth=0.0,
#              color="red",
#              label="Left tail",
#          )
#      z.xticks(range(1, len(key) + 1, 1), key)
#      z.xlim(xmin=0.5, xmax=len(key) + 0.5)
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
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=2, mode="expand", borderaxespad=0
#      )
#      z.grid()
#      z.show()
#
#      if tail_selected == "Right" or tail_selected == "Both":
#
#          # Confidence interval for the right tail
#          z.figure("Confidence interval for the right tail")
#          z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#          z.plot(
#              range(1, len(key) + 1, 1),
#              positive_alpha_vec,
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="green",
#              label="Right tail",
#          )
#          z.plot(
#              range(1, len(key) + 1, 1),
#              positive_upper_bound,
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="purple",
#              label="Upper bound",
#          )
#          z.plot(
#              range(1, len(key) + 1, 1),
#              positive_lower_bound,
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="blue",
#              label="Lower bound",
#          )
#          z.plot(range(0, len(key) + 2, 1), np.repeat(3, len(key) + 2), color="red")
#          z.plot(range(0, len(key) + 2, 1), np.repeat(2, len(key) + 2), color="red")
#          z.xticks(range(1, len(key) + 1, 1), key)
#          z.xlim(xmin=0.5, xmax=len(key) + 0.5)
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
#          z.show()
#
#      if tail_selected == "Left" or tail_selected == "Both":
#
#          # Confidence interval for the left tail
#          z.figure("Confidence interval for the left tail")
#          z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#          z.plot(
#              range(1, len(key) + 1, 1),
#              negative_alpha_vec,
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="green",
#              label="Left tail",
#          )
#          z.plot(
#              range(1, len(key) + 1, 1),
#              negative_upper_bound,
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="purple",
#              label="Upper bound",
#          )
#          z.plot(
#              range(1, len(key) + 1, 1),
#              negative_lower_bound,
#              marker="o",
#              markersize=7.0,
#              linewidth=0.0,
#              color="blue",
#              label="Lower bound",
#          )
#          z.plot(range(0, len(key) + 2, 1), np.repeat(3, len(key) + 2), color="red")
#          z.plot(range(0, len(key) + 2, 1), np.repeat(2, len(key) + 2), color="red")
#          z.xticks(range(1, len(key) + 1, 1), key)
#          z.xlim(xmin=0.5, xmax=len(key) + 0.5)
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
#          z.show()
#
#      # Absolute length of the tail bar chart
#
#      z.figure("Absolute tail lengths")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      amplitude = 0.5
#      if tail_selected == "Right" or tail_selected == "Both":
#          z.bar(
#              np.arange(0, 2 * len(key), 2),
#              positive_abs_length,
#              amplitude,
#              color="green",
#              label="Right tail",
#          )
#      if tail_selected == "Left" or tail_selected == "Both":
#          z.bar(
#              np.arange(amplitude, 2 * len(key) + amplitude, 2),
#              negative_abs_length,
#              amplitude,
#              color="red",
#              label="Left tail",
#          )
#      z.xticks(np.arange(amplitude, 2 * len(key) + amplitude, 2), key)
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
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3, mode="expand", borderaxespad=0
#      )
#      z.grid()
#      z.show()
#
#      # Absolute length of the tail bar chart
#
#      z.figure("Relative tail lengths")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      amplitude = 0.5
#      if tail_selected == "Right" or tail_selected == "Both":
#          z.bar(
#              np.arange(0, 2 * len(key), 2),
#              positive_rel_length,
#              amplitude,
#              color="green",
#              label="Right tail",
#          )
#      if tail_selected == "Left" or tail_selected == "Both":
#          z.bar(
#              np.arange(amplitude, 2 * len(key) + amplitude, 2),
#              negative_rel_length,
#              amplitude,
#              color="red",
#              label="Left tail",
#          )
#      z.xticks(np.arange(amplitude, 2 * len(key) + amplitude, 2), key)
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
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3, mode="expand", borderaxespad=0
#      )
#      z.grid()
#      z.show()
#
#      # KS test outcome
#
#      z.figure("KS test p value for the tails")
#      z.gca().set_position((0.1, 0.20, 0.83, 0.70))
#      amplitude = 0.5
#      if tail_selected == "Right" or tail_selected == "Both":
#          z.bar(
#              np.arange(0, 2 * len(key), 2),
#              positive_alpha_KS,
#              amplitude,
#              color="green",
#              label="Right tail",
#          )
#      if tail_selected == "Left" or tail_selected == "Both":
#          z.bar(
#              np.arange(amplitude, 2 * len(key) + amplitude, 2),
#              negative_alpha_KS,
#              amplitude,
#              color="red",
#              label="Left tail",
#          )
#      z.xticks(np.arange(amplitude, 2 * len(key) + amplitude, 2), key)
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
#          bbox_to_anchor=(0.0, -0.175, 1.0, 0.02), ncol=3, mode="expand", borderaxespad=0
#      )
#      z.grid()
#      z.show()
#
#      # Print the figures
#      matrix_form = np.array(tail_statistics)
#      matrix_form_transpose = np.transpose(matrix_form)
#      filename = "TailStatistics_Overall.csv"
#      df = pd.DataFrame(
#          {
#              "Input": key,
#              "Positive Tail Exponent": matrix_form_transpose[0],
#              "Negative Tail Exponent": matrix_form_transpose[1],
#              "Positive Tail xmin": matrix_form_transpose[2],
#              "Negative Tail xmin": matrix_form_transpose[3],
#              "Positive Tail S.Err": matrix_form_transpose[4],
#              "Negative Tail S.Err": matrix_form_transpose[5],
#              "Positive Tail Size": matrix_form_transpose[6],
#              "Negative Tail Size": matrix_form_transpose[7],
#              "Positive Tail KS p-value": matrix_form_transpose[8],
#              "Negative Tail KS p-value": matrix_form_transpose[9],
#          }
#      )
#
#      df = df[
#          [
#              "Input",
#              "Positive Tail Exponent",
#              "Negative Tail Exponent",
#              "Positive Tail xmin",
#              "Negative Tail xmin",
#              "Positive Tail S.Err",
#              "Negative Tail S.Err",
#              "Positive Tail Size",
#              "Negative Tail Size",
#              "Positive Tail KS p-value",
#              "Negative Tail KS p-value",
#          ]
#      ]
#      df.to_csv(filename, index=False)


#  elif approach == "Rolling" or approach == "Increasing":
if approach == "Rolling" or approach == "Increasing":

    #  question      = "Do you want to save the sequential scaling plot?"
    #  choices      = ['Yes', 'No']
    #  plot_storing = eg.choicebox(question, 'Plot', choices)
    plot_storing = "No"

    #  if plot_storing == "Yes":
    #      question = "What is the target directory for the pictures?"
    #      motherpath = eg.enterbox(
    #          question,
    #          title="path",
    #          default="C:\Users\\alber\Dropbox\Research\IP\Econophysics\Final Code Hurst Exponent\\",
    #      )

    initial_index = database[0].index(initial_date)
    final_index = database[0].index(final_date)
    dates = database[0][initial_index: (final_index + 1)]
    labelstep = (22 if len(dates) <= 252 else
                 66 if (len(dates) > 252 and len(dates) <= 756) else
                 121)
    N = len(database)

    filtered_dates = []

    tail_statistics = {}

    results = results_dicts_init()
    xmin_vecs = xmin_lists_init()

    for l in range(initial_index, final_index + 1, sliding_window):

        BlockDict = {}

        if approach == "Rolling":
            begin_date = database[0][(l + 1 - lookback)]
            end_date = database[0][l]
        else:
            begin_date = database[0][(initial_index + 1 - lookback)]
            end_date = database[0][l]

        filtered_dates.append(end_date)

        print(
            "I am analyzing the time series between " + begin_date + " and " + end_date
        )

        for i in range(1, N, 1):

            if approach == "Rolling":
                series = database[i][(l + 1 - lookback) : (l + 1)]
            else:
                series = database[i][(initial_index + 1 - lookback) : (l + 1)]

            series = np.array(series)

            print(
                "You opted for the analysis of the "
                + input_type
                + " under a "
                + approach
                + " time window."
            )
            print(
                "Your lookback is "
                + str(lookback)
                + " days and you are advancing in time with a step of "
                + str(sliding_window)
                + " days."
            )
            print(
                "Currently two consecutive time series have an overlap of "
                + str(np.maximum(0, lookback - sliding_window))
                + " observations."
            )

            X = utils.preprocess_series(series)

            # NOTE: standardization & absolutization conditional
            #       below slightly different
            #  if standardize == "Yes":
            #      if norm_timing == "Before" or norm_timing == "Both":
            #          print("I am standardizing your time series")
            #          S = X
            #          m = np.mean(S)
            #          v = np.std(S)
            #          X = (S - m) / v
            #  if abs_value == "Yes":
            #      if abs_timing == "Before" or abs_timing == "Both":
            #          print("I am taking the absolute value of your time series")
            #          X = np.abs(X)

            # TODO: use defaultdict, np.hstack(?), ...
            if identifiers[i - 1] in BlockDict:
                X0 = BlockDict[identifiers[i-1]]
                print(
                    "I found an existing group under the identifier : "
                    + identifiers[i - 1]
                    + ". Your time series is added in that pool"
                )
                BlockDict[identifiers[i - 1]] = np.hstack((X0, X))
            else:
                print(
                    "I have not found an existing group under the identifier : "
                    + identifiers[i - 1]
                    + ". I create the group and i add your time series is added in that pool"
                )
                BlockDict[identifiers[i - 1]] = X  # NOTE: keep X as NumPy array

        key = BlockDict.keys()
        for el in key:
            print(
                "I am analyzing the group identified by "
                + el
                + " between "
                + begin_date
                + " and "
                + end_date
            )

            X = BlockDict[el]

            #  if standardize == "Yes":
            #      if norm_timing == "After" or norm_timing == "Both":
            #          print("I am standardizing your time series")
            #          S = X
            #          m = np.mean(S)
            #          v = np.std(S)
            #          X = (S - m) / v
            #  if abs_value == "Yes":
            #      if abs_timing == "After" or abs_timing == "Both":
            #          print("I am taking the absolute value of your time series")
            #          X = np.abs(X)

            #  X = X[X != 0]  # NOTE: only keep/use non-zero elements

            if s.use_right_tail:
                tail_plus, fit1 = utils.fit_tail(X)
            if s.use_left_tail:
                tail_neg, fit2 = utils.fit_tail(-X)

            if data_nature == "Continuous":

                if s.use_right_tail:
                    xmin_vecs["xmin_vec_right"].append(xmin_r0 := fit1.power_law.xmin)
                if s.use_left_tail:
                    xmin_vecs["xmin_vec_left"].append(xmin_l0 := fit2.power_law.xmin)

                # TODO: integrate Rolling xmin_rule into utils module
                if xmin_rule == "Rolling":
                    roll_tot = rolling_days + rolling_lags

                    if s.use_right_tail:
                        if len(xmin_vecs["xmin_vec_right"]) < roll_tot:
                            fit_1 = utils.powerlaw_fit(X)
                        else:
                            xmin_avg = np.average(xmin_vecs["xmin_vec_right"]
                                                  [-roll_tot:-rolling_lags])
                            fit_1 = utils.powerlaw_fit(X, xmin=xmin_avg)

                    if s.use_left_tail:
                        if len(xmin_vecs["xmin_vec_left"]) < roll_tot:
                            fit_2 = utils.powerlaw_fit(-X)
                        else:
                            xmin_avg = np.average(xmin_vecs["xmin_vec_left"]
                                                  [-roll_tot:-rolling_lags])
                            fit_2 = utils.powerlaw_fit(-X, xmin=xmin_avg)
                else:  # NOTE: Clauset, Manual & Percentile
                    # TODO: when Clauset, just use already calculated fit objects above
                    if s.use_right_tail:
                        fit_1 = utils.powerlaw_fit(X)
                    if s.use_left_tail:
                        fit_2 = utils.powerlaw_fit(-X)

            #  elif data_nature == "Discrete":
            #      if tail_selected == "Right" or tail_selected == "Both":
            #          xmin_today_right = (
            #              pl.Fit(list(filter(lambda x: x != 0, tail_plus), discrete=True))
            #          ).power_law.xmin
            #          xmin_vecs["xmin_vec_right"].append(xmin_today_right)
            #      if tail_selected == "Left" or tail_selected == "Both":
            #          xmin_today_left = (
            #              pl.Fit(list(filter(lambda x: x != 0, tail_neg), discrete=True))
            #          ).power_law.xmin
            #          xmin_vecs["xmin_vec_left"].append(xmin_today_left)
            #
            #      if xmin_rule == "Clauset":
            #          if tail_selected == "Right" or tail_selected == "Both":
            #              fit_1 = pl.Fit(
            #                  list(filter(lambda x: x != 0, tail_plus), discrete=True)
            #              )
            #          if tail_selected == "Left" or tail_selected == "Both":
            #              fit_2 = pl.Fit(
            #                  list(filter(lambda x: x != 0, tail_neg), discrete=True)
            #              )
            #      elif xmin_rule == "Rolling":
            #          if tail_selected == "Right" or tail_selected == "Both":
            #              if len(xmin_vecs["xmin_vec_right"]) < rolling_days + rolling_lags:
            #                  fit_1 = pl.Fit(
            #                      list(filter(lambda x: x != 0, tail_plus), discrete=True)
            #                  )
            #              else:
            #                  avg_xmin = np.average(
            #                      xmin_vecs["xmin_vec_right"][
            #                          -(rolling_days + rolling_lags) : -(rolling_lags)
            #                      ]
            #                  )
            #                  fit_1 = pl.Fit(
            #                      list(filter(lambda x: x != 0, tail_plus)),
            #                      discrete=True,
            #                      xmin=avg_xmin,
            #                  )
            #          if tail_selected == "Left" or tail_selected == "Both":
            #              if len(xmin_vecs["xmin_vec_left"]) < rolling_days + rolling_lags:
            #                  fit_2 = pl.Fit(
            #                      list(filter(lambda x: x != 0, tail_neg), discrete=True)
            #                  )
            #              else:
            #                  avg_xmin = np.average(
            #                      xmin_vecs["xmin_vec_left"][
            #                          -(rolling_days + rolling_lags) : -(rolling_lags)
            #                      ]
            #                  )
            #                  fit_2 = pl.Fit(
            #                      list(filter(lambda x: x != 0, tail_neg)),
            #                      discrete=True,
            #                      xmin=avg_xmin,
            #                  )
            #
            #      elif xmin_rule == "Manual":
            #          if tail_selected == "Right" or tail_selected == "Both":
            #              fit_1 = pl.Fit(
            #                  list(filter(lambda x: x != 0, tail_plus)),
            #                  discrete=True,
            #                  xmin=xmin_value,
            #              )
            #          if tail_selected == "Left" or tail_selected == "Both":
            #              fit_2 = pl.Fit(
            #                  list(filter(lambda x: x != 0, tail_neg)),
            #                  discrete=True,
            #                  xmin=xmin_value,
            #              )
            #      else:
            #          if tail_selected == "Right" or tail_selected == "Both":
            #              fit_1 = pl.Fit(
            #                  list(filter(lambda x: x != 0, tail_plus)),
            #                  discrete=True,
            #                  xmin=np.percentile(tail_plus, xmin_sign),
            #              )
            #          if tail_selected == "Left" or tail_selected == "Both":
            #              fit_2 = pl.Fit(
            #                  list(filter(lambda x: x != 0, tail_neg)),
            #                  discrete=True,
            #                  xmin=np.percentile(tail_neg, xmin_sign),
            #              )

            if tail_selected == "Right" or tail_selected == "Both":

                alpha1 = fit_1.power_law.alpha
                xmin1 = fit_1.power_law.xmin
                s_err1 = fit_1.power_law.sigma
                p1 = plpva.plpva(
                    np.array(tail_plus).tolist(), float(xmin1), "reps", c_iter, "silent"
                )
                if el in results["pos_α_vec"]:
                    results["pos_α_vec"][el].append(alpha1)
                    results["pos_up_bound"][el].append(
                        alpha1 + (st.norm.ppf(1 - multiplier * significance)) * s_err1
                    )
                    results["pos_low_bound"][el].append(
                        alpha1 - (st.norm.ppf(1 - multiplier * significance)) * s_err1
                    )
                    results["pos_abs_len"][el].append(
                        len(list(filter(lambda x: x >= xmin1, tail_plus)))
                    )
                    results["pos_rel_len"][el].append(
                        len(list(filter(lambda x: x >= xmin1, tail_plus)))
                        / float(len(tail_plus))
                    )
                    results["pos_α_ks"][el].append(p1[0])
                else:
                    results["pos_α_vec"][el] = [alpha1]
                    results["pos_up_bound"][el] = [
                        alpha1 + (st.norm.ppf(1 - multiplier * significance)) * s_err1
                    ]
                    results["pos_low_bound"][el] = [
                        alpha1 - (st.norm.ppf(1 - multiplier * significance)) * s_err1
                    ]
                    results["pos_abs_len"][el] = [
                        len(list(filter(lambda x: x >= xmin1, tail_plus)))
                    ]
                    results["pos_rel_len"][el] = [
                        len(list(filter(lambda x: x >= xmin1, tail_plus)))
                        / float(len(tail_plus))
                    ]
                    results["pos_α_ks"][el] = [p1[0]]

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

                if el in results["loglr_right"]:
                    results["loglr_right"][el].append(daily_r_ratio)
                    results["loglpv_right"][el].append(daily_r_p)
                else:
                    results["loglr_right"][el] = [daily_r_ratio]
                    results["loglpv_right"][el] = [daily_r_p]

            if tail_selected == "Left" or tail_selected == "Both":

                alpha2 = fit_2.power_law.alpha
                xmin2 = fit_2.power_law.xmin
                s_err2 = fit_2.power_law.sigma
                p2 = plpva.plpva(
                    np.array(tail_neg).tolist(), float(xmin2), "reps", c_iter, "silent"
                )
                if el in results["neg_α_vec"]:
                    results["neg_α_vec"][el].append(alpha2)
                    results["neg_up_bound"][el].append(
                        alpha2 + (st.norm.ppf(1 - multiplier * significance)) * s_err2
                    )
                    results["neg_low_bound"][el].append(
                        alpha2 - (st.norm.ppf(1 - multiplier * significance)) * s_err2
                    )
                    results["neg_abs_len"][el].append(
                        len(list(filter(lambda x: x >= xmin2, tail_neg)))
                    )
                    results["neg_rel_len"][el].append(
                        len(list(filter(lambda x: x >= xmin2, tail_neg)))
                        / float(len(tail_neg))
                    )
                    results["neg_α_ks"][el].append(p2[0])
                else:
                    results["neg_α_vec"][el] = [alpha2]
                    results["neg_up_bound"][el] = [
                        alpha2 + (st.norm.ppf(1 - multiplier * significance)) * s_err2
                    ]
                    results["neg_low_bound"][el] = [
                        alpha2 - (st.norm.ppf(1 - multiplier * significance)) * s_err2
                    ]
                    results["neg_abs_len"][el] = [
                        len(list(filter(lambda x: x >= xmin2, tail_neg)))
                    ]
                    results["neg_rel_len"][el] = [
                        len(list(filter(lambda x: x >= xmin2, tail_neg)))
                        / float(len(tail_neg))
                    ]
                    results["neg_α_ks"][el] = [p2[0]]

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

                if el in results["loglr_left"]:
                    results["loglr_left"][el].append(daily_l_ratio)
                    results["loglpv_left"][el].append(daily_l_p)
                else:
                    results["loglr_left"][el] = [daily_l_ratio]
                    results["loglpv_left"][el] = [daily_l_p]

            #  if plot_storing == "Yes":
            #
            #      directory = (
            #          motherpath
            #          + "PowerLawAnimation\\"
            #          + el
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
            #          os.makedirs(directory)
            #      except OSError:
            #          if not os.path.isdir(directory):
            #              raise
            #      os.chdir(directory)
            #
            #      if tail_selected == "Right" or tail_selected == "Both":
            #
            #          plt.figure("Right tail scaling for group " + el)
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
            #              "Log-log plot of the scaling properties of the right-tail for the group "
            #              + el
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
            #                  len(list(filter(lambda x: x > xmin1, tail_plus)),),
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
            #              + el
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #          plt.figure("Right tail comparison for group " + el)
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
            #              "Comparison of the distributions fitted on the right-tail for the group "
            #              + el
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
            #              + el
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #      if tail_selected == "Left" or tail_selected == "Both":
            #
            #          plt.figure("Left tail scaling for group " + el)
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
            #              "Log-log plot of the scaling properties of the left-tail for group "
            #              + el
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
            #                  len(list(filter(lambda x: x > xmin2, tail_neg)),),
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
            #              + el
            #              + ".jpg"
            #          )
            #          plt.close()
            #
            #          plt.figure("Left tail comparison for group " + el)
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
            #              "Comparison of the distributions fitted on the left-tail for the group "
            #              + el
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
            #              + el
            #              + ".jpg"
            #          )
            #          plt.close()

            if tail_selected == "Both":
                row = [
                    alpha1,
                    alpha2,
                    xmin1,
                    xmin2,
                    xmin_r0,
                    xmin_l0,
                    s_err1,
                    s_err2,
                    len(list(filter(lambda x: x >= xmin1, tail_plus))),
                    len(list(filter(lambda x: x >= xmin2, tail_neg))),
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

            #  if tail_selected == "Right":
            #      row = [
            #          alpha1,
            #          0,
            #          xmin1,
            #          0,
            #          xmin_today_right,
            #          0,
            #          s_err1,
            #          0,
            #          len(list(filter(lambda x: x >= xmin1, tail_plus))),
            #          0,
            #          p1[0],
            #          0,
            #          daily_r_ratio[0],
            #          daily_r_ratio[1],
            #          daily_r_ratio[2],
            #          daily_r_p[0],
            #          daily_r_p[1],
            #          daily_r_p[2],
            #          0,
            #          0,
            #          0,
            #          0,
            #          0,
            #          0,
            #      ]
            #  if tail_selected == "Left":
            #      row = [
            #          0,
            #          alpha2,
            #          0,
            #          xmin2,
            #          0,
            #          xmin_today_left,
            #          0,
            #          s_err2,
            #          0,
            #          len(list(filter(lambda x: x >= xmin2, tail_neg))),
            #          0,
            #          p2[0],
            #          0,
            #          0,
            #          0,
            #          0,
            #          0,
            #          0,
            #          daily_l_ratio[0],
            #          daily_l_ratio[1],
            #          daily_l_ratio[2],
            #          daily_l_p[0],
            #          daily_l_p[1],
            #          daily_l_p[2],
            #      ]

            if el in tail_statistics:
                tail_statistics[el].append(row)
            else:
                tail_statistics[el] = [row]

    ## Cross clusters analysis

    if tail_selected == "Right" or tail_selected == "Both":
        max_key, max_value = max(
            results["pos_α_vec"].items(), key=lambda x: len(set(x[1]))
        )

        z.figure("Cross sectional analysis for the positive tail")
        z.gca().set_position((0.1, 0.20, 0.83, 0.70))
        key = BlockDict.keys()
        for el in key:
            z.plot(
                np.arange(0.5, len(results["pos_α_vec"][el]) + 0.5, 1),
                results["pos_α_vec"][el],
                marker="o",
                label=el,
            )
        z.plot(np.repeat(3, len(results["pos_α_vec"][max_key]) + 1), color="red")
        z.plot(np.repeat(2, len(results["pos_α_vec"][max_key]) + 1), color="red")
        z.ylabel(r"$\alpha$")
        z.xlim(xmin=0.0, xmax=len(results["pos_α_vec"][max_key]))
        z.xticks(
            range(
                0,
                len(results["pos_α_vec"][max_key]),
                int(np.maximum(int(labelstep / float(sliding_window)), 1)),
            ),
            [
                elem[3:]
                for elem in filtered_dates[0 :: int(labelstep / float(sliding_window))]
            ],
            rotation="vertical",
        )
        z.title(
            "Cross-sectional analysis for "
            + r"$\alpha$"
            + "-right tail exponents "
            + "\n"
            + "Time Period: "
            + dates[0]
            + " - "
            + dates[len(dates) - 1]
            + ". Grouping rule: "
            + partition
        )
        z.legend()
        z.grid()
        #  z.show()

        # Plotting the histograms for the rolling alpha

        key = BlockDict.keys()
        for el in key:
            z.figure(
                "Histogram of positive tail alphas for " + el, figsize=(8, 6), dpi=100
            )
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            IQR = np.percentile(results["pos_α_vec"][el], 75) - np.percentile(
                results["pos_α_vec"][el], 25
            )
            h = 2 * IQR * np.power(len(results["pos_α_vec"][el]), -1.0 / 3.0)
            nbins = np.int(
                (np.max(results["pos_α_vec"][el]) - np.min(results["pos_α_vec"][el]))
                / float(h)
            )
            # Building the histogram and plotting the relevant vertical lines
            z.hist(results["pos_α_vec"][el], nbins, color="red")
            out1, bins = z.histogram(results["pos_α_vec"][el], nbins)
            z.plot(
                np.repeat(np.mean(results["pos_α_vec"][el]), np.max(out1) + 1),
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
                + el
                + "\n"
                + "Time period: "
                + dates[0]
                + " - "
                + dates[-1]
            )
            z.xlim(
                xmin=np.min(results["pos_α_vec"][el]), xmax=np.max(results["pos_α_vec"][el])
            )
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
                    np.round(np.mean(results["pos_α_vec"][el]), 4),
                    np.round(np.std(results["pos_α_vec"][el]), 4),
                    np.round(np.min(results["pos_α_vec"][el]), 4),
                    np.round(np.max(results["pos_α_vec"][el]), 4),
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
            #  z.show()

    if tail_selected == "Left" or tail_selected == "Both":
        max_key, max_value = max(
            results["neg_α_vec"].items(), key=lambda x: len(set(x[1]))
        )

        z.figure("Cross sectional analysis for the negative tail")
        z.gca().set_position((0.1, 0.20, 0.83, 0.70))
        key = BlockDict.keys()
        for el in key:
            z.plot(
                np.arange(0.5, len(results["neg_α_vec"][el]) + 0.5, 1),
                results["neg_α_vec"][el],
                marker="o",
                label=el,
            )
        z.plot(np.repeat(3, len(results["neg_α_vec"][max_key]) + 1), color="red")
        z.plot(np.repeat(2, len(results["neg_α_vec"][max_key]) + 1), color="red")
        z.ylabel(r"$\alpha$")
        z.xlim(xmin=0.0, xmax=len(results["neg_α_vec"][max_key]))
        z.xticks(
            range(
                0,
                len(results["neg_α_vec"][max_key]),
                int(labelstep / float(sliding_window)),
            ),
            [
                elem[3:]
                for elem in filtered_dates[0 :: int(labelstep / float(sliding_window))]
            ],
            rotation="vertical",
        )
        z.title(
            "Cross-sectional analysis for "
            + r"$\alpha$"
            + "-left tail exponents "
            + "\n"
            + "Time Period: "
            + dates[0]
            + " - "
            + dates[len(dates) - 1]
            + ". Grouping rule: "
            + partition
        )
        z.legend()
        z.grid()
        #  z.show()

        # Plotting the histograms for the rolling alpha

        key = BlockDict.keys()
        for el in key:
            z.figure(
                "Histogram of negative tail alphas for " + el, figsize=(8, 6), dpi=100
            )
            z.gca().set_position((0.1, 0.20, 0.83, 0.70))
            IQR = np.percentile(results["neg_α_vec"][el], 75) - np.percentile(
                results["neg_α_vec"][el], 25
            )
            h = 2 * IQR * np.power(len(results["neg_α_vec"][el]), -1.0 / 3.0)
            nbins = np.int(
                (np.max(results["neg_α_vec"][el]) - np.min(results["neg_α_vec"][el]))
                / float(h)
            )
            # Building the histogram and plotting the relevant vertical lines
            z.hist(results["neg_α_vec"][el], nbins, color="red")
            out1, bins = z.histogram(results["neg_α_vec"][el], nbins)
            z.plot(
                np.repeat(np.mean(results["neg_α_vec"][el]), np.max(out1) + 1),
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
                + el
                + "\n"
                + "Time period: "
                + dates[0]
                + " - "
                + dates[-1]
            )
            z.xlim(
                xmin=np.min(results["neg_α_vec"][el]), xmax=np.max(results["neg_α_vec"][el])
            )
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
                    np.round(np.mean(results["neg_α_vec"][el]), 4),
                    np.round(np.std(results["neg_α_vec"][el]), 4),
                    np.round(np.min(results["neg_α_vec"][el]), 4),
                    np.round(np.max(results["neg_α_vec"][el]), 4),
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
            #  z.show()

    for el in tail_statistics.keys():
        # Print the figures
        matrix_form = np.array(tail_statistics[el])
        matrix_form_transpose = np.transpose(matrix_form)
        filename = "TailStatistics_" + el + "MA_KS_d=66.csv"
        df = pd.DataFrame(
            {
                "Date": filtered_dates,
                "Positive Tail Exponent": matrix_form_transpose[0],
                "Negative Tail Exponent": matrix_form_transpose[1],
                "Positive Tail xmin used": matrix_form_transpose[2],
                "Negative Tail xmin used": matrix_form_transpose[3],
                "Positive Tail xmin today": matrix_form_transpose[4],
                "Negative Tail xmin today": matrix_form_transpose[5],
                "Positive Tail S.Err": matrix_form_transpose[6],
                "Negative Tail S.Err": matrix_form_transpose[7],
                "Positive Tail Size": matrix_form_transpose[8],
                "Negative Tail Size": matrix_form_transpose[9],
                "Positive Tail KS p-value": matrix_form_transpose[10],
                "Negative Tail KS p-value": matrix_form_transpose[11],
                "LL Ratio Right Tail TPL": matrix_form_transpose[12],
                "LL Ratio Right Tail Exp": matrix_form_transpose[13],
                "LL Ratio Right Tail LogN": matrix_form_transpose[14],
                "LL p-value Right Tail TPL": matrix_form_transpose[15],
                "LL p-value Right Tail Exp": matrix_form_transpose[16],
                "LL p-value Right Tail LogN": matrix_form_transpose[17],
                "LL Ratio Left Tail TPL": matrix_form_transpose[18],
                "LL Ratio Left Tail Exp": matrix_form_transpose[19],
                "LL Ratio Left Tail LogN": matrix_form_transpose[20],
                "LL p-value Left Tail TPL": matrix_form_transpose[21],
                "LL p-value Left Tail Exp": matrix_form_transpose[22],
                "LL p-value Left Tail LogN": matrix_form_transpose[23],
            }
        )

        df = df[
            [
                "Date",
                "Positive Tail Exponent",
                "Negative Tail Exponent",
                "Positive Tail xmin used",
                "Negative Tail xmin used",
                "Positive Tail xmin today",
                "Negative Tail xmin today",
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
