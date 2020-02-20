from types import SimpleNamespace

from ui import get_uis


uis = get_uis.main(standalone_mode=False)
settings = SimpleNamespace(**uis)

#  database_name = "dbMSTR_test.csv"
#
#  #  no_entries = 1
#  #  fieldNames = ["# " + str(i) for i in range(1, no_entries + 1, 1)]
#  tickers = ["DE 01Y"]  # , "DE 03Y", "DE 05Y", "DE 10Y"]
#
#  database = pd.read_csv(database_name, index_col="Date")[tickers]
#  N_db_rec, N_db_tck = database.shape
#  assert(N_db_rec == 3333)
#  assert(N_db_tck == len(tickers))
#
#  db_dates = database.index
#
#  # FIXME?: using date below -> ValueError: cannot convert float NaN to integer
#  #  initial_date = "2/5/2016"  # NOTE: len(dates) needs to be > labelstep???
#  #  date_i = "1/4/2016"
#  date_i = "31-03-16"
#  #  initial_date = "1/1/2016"
#  date_f = "5/5/2016"
#  # TODO: standardize/validate date format
#  # TODO: consider allow for free-forming date range, then pick closest dates
#  lookback = 504
#
#  ind_i = db_dates.get_loc(date_i)
#  ind_f = db_dates.get_loc(date_f)
#  n_vec = ind_f - ind_i + 1
#  dates = db_dates[ind_i: ind_f + 1]
#  assert(len(dates) == n_vec)
#
#  labelstep = (22 if n_vec <= 252 else
#               66 if (n_vec > 252 and n_vec <= 756) else
#               121)
#
#  # NOTE: data here does not contain values needed in lookback
#  # TODO: better name might be dates_analyzed_df
#  ticker_df = database.iloc[ind_i: ind_f + 1]
#  assert((n_vec, len(tickers)) == ticker_df.shape)
#
#  #  N = len(database)
#  #  for l in range(initial_index, final_index + 1, anal_freq):
#
#  return_type = "log"  # choices is one of ["basic", "relative", "log"]
#
#  tau = 1
#
#  standardize = "No"
#  standardize_target = "Tail"  # choices is one of ['Full Series', 'Tail']
#
#  abs_value = "No"
#  abs_target = "Tail"  # choices is one of ['Full Series', 'Tail']
#
#  approach = "Rolling"  # choices is one of ['Static', 'Rolling', 'Increasing']
#  anal_freq = 1
#
#  tail_selected = "Both"
#  use_right_tail = True if tail_selected in ["Right", "Both"] else False
#  use_left_tail = True if tail_selected in ["Left", "Both"] else False
#  if tail_selected == "Both":
#      multiplier = 0.5
#  else:
#      multiplier = 1.0
#
#  data_nature = "Continuous"
#
#  xmin_rule = "Clauset"
#  xmin_value = None  # only used for xmin_rule == "Manual"
#  xmin_sign = None  # only used for xmin_rule == "Percentile"
#
#  significance = 0.05
#
#  c_iter = 100
#
#  # NOTE: if anal_freq == 1, then dates also == dates[::anal_freq]
#  spec_dates = dates[::anal_freq] if anal_freq > 1 else dates
#  #  n_spdt = len(spec_dates)
#  spec_labelstep = 22 if anal_freq > 1 else labelstep
#
#  show_plots = True
#  save_plots = False
#
#
#  # object to hold all options data determined by user input data
#  # NOTE: consider using json (module), dataclasses, namedtuple?
#  # TODO: set values of these dynamically based on user input
#  settings_dict = {"tickers": tickers,
#                   "lookback": lookback,
#                   "return_type": return_type,
#                   "tau": tau,
#                   "standardize": False,
#                   "STDZ_TARGET": "Tail",  # FIXME: use better primitive
#                   "absolutize": False,
#                   "ABSZ_TARGET": "Tail",  # FIXME: use better primitive
#                   "approach": approach,
#                   # NOTE: anal_freq only defined for approach != 'Static'
#                   "anal_freq": anal_freq,
#                   "use_right_tail": use_right_tail,
#                   "use_left_tail": use_left_tail,
#                   "data_nature": data_nature,
#                   "xmin_rule": xmin_rule,
#                   "XMIN_VALUE": xmin_value,
#                   "XMIN_SIGN": xmin_sign,
#                   "significance": significance,
#                   "dates": dates,
#                   "date_i": dates[0],
#                   "date_f": dates[-1],
#                   "n_vec": n_vec,  # FIXME: should be length of spec_dates?
#                   "labelstep": labelstep,
#                   "spec_dates": spec_dates,
#                   "N_SPDT": len(spec_dates),
#                   "spec_labelstep": spec_labelstep,
#                   "show_plots": show_plots,
#                   "save_plots": save_plots}
# TODO: add "labels" and other important values into options dict
#  settings = SimpleNamespace(**settings_dict)
