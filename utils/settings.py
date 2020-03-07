from statistics import NormalDist


def _get_tails_used(tail_selection):
    """Return relevant tail selection settings
    """
    use_right = True if tail_selection in ('right', 'both') else False
    use_left = True if tail_selection in ('left', 'both') else False

    tails_used = []
    if use_right:
        tails_used.append("right")
    if use_left:
        tails_used.append("left")

    return use_right, use_left, tuple(tails_used)


def get_xmin(xmin_args):
    rule, *vqarg = xmin_args
    # check that additional arg(s) given after xmin_rule is/are numeric(s)
    if not all(map(str.isdecimal, vqarg)):
        raise ValueError(f"extra arg(s) to xmin rule '{rule}' must be "
                         f"numeric type(s), given: {', '.join(vqarg)}")
#  def xmin_cb(ctx, param, xmin):
#      print('fired xmin callback')
#      if len(xmin) == 2:
#          return
#      elif len(xmin) == 1:
#          #  if xmin[0] == 'clauset':
#          #      return ('clauset', None)
#          #  elif xmin[0] == 'manual':
#          #      return ('manual', 0)
#          #  elif xmin[0] == 'percentile':
#          #      return ('percentile', 90)
#          xmin_defaults = {'clauset': None, 'manual': 0, 'percentile': 90}
#          return xmin, xmin_defaults[xmin]


# TODO: make distinction b/w private/internal & public setting?
# TODO: OR distinguish b/w analysis vs. control-flow settings!!
def set_context(kwd):
    #  print(f'using tickers: {tickers}')
    #
    #  db_df = pd.read_csv(db_file, index_col='Date')[tickers]
    #  db_dates = db_df.index
    #  ind_i, ind_f = db_dates.get_loc(date_i), db_dates.get_loc(date_f)
    #  n_vec = ind_f - ind_i + 1  # FIXME: should be length of spec_dates?
    #  dates = db_dates[ind_i: ind_f + 1]
    #
    #  labelstep = (22 if n_vec <= 252 else
    #               66 if (n_vec > 252 and n_vec <= 756) else
    #               121)
    #
    #  # TODO: remove need for "spec_" variables
    #  if anal_freq > 1:
    #      spec_dates = dates[::anal_freq]
    #      spec_labelstep = 22
    #  elif anal_freq == 1:
    #      spec_dates = dates
    #      spec_labelstep = labelstep
    #  n_spdt = len(spec_dates)
    #
    #  ticker_df = db_df.iloc[ind_i: ind_f + 1]
    #
    #  use_right_tail, use_left_tail, tails_used = get_tails_used(tail_select)
    #
    #  _tail_mult = 0.5 if tail_selected == 'both' else 1
    #  alpha_quantile = NormalDist().inv_cdf(1 - _tail_mult * alpha_sgnf)

    #  if xmin_rule != 'clauset' and xmin_var_qty is None:
    #      xmin_var_qty = prompt_xmin_var_qty(xmin_rule)

    #  click.echo(locals())

    #  return locals()
    pass
