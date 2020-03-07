import click

from dct_cbs import attach_script_opts, gset_db_df, gset_group_opts

#  from statistics import NormalDist


#  def _get_db_choices():
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


# TODO/TODO: update conda channel(s) to Click 7.1 for show_default kwarg
@click.command(  # name='',
               context_settings=dict(default_map=None,
                                     max_content_width=100,  # TODO: use 120?
                                     help_option_names=('-h', '--help'),
                                     #  token_normalize_func=None,
                                     #  show_default=True,
                                     ),
               epilog='')
# TODO: customize --help
#   - widen first help column of options/args --> HelpFormatter.write_dl()
#   - better formatting/line wrapping for options of type click.Choice
#   - option help texts when multiline, doesn't wrap properly
#   - remove cluttering & useless type annotations (set options' metavar attr)
@click.argument('db_df', metavar='DB_FILE', nargs=1, is_eager=True,
                type=click.File(mode='r'), callback=gset_db_df,
                default='dbMSTR_test.csv')  # TODO: allow 2nd pos-arg for cfg?
# TODO: consider moving DB_FILE & -G into config/options/attributes.yaml ??
@click.option('-G', '--group/--no-group', 'analyze_group',
              is_eager=True, callback=gset_group_opts,
              default=False, show_default=True,
              help=('set flag to run in group analysis mode; use with --help '
                    'to also see options specific to group analysis'))
@attach_script_opts()  # NOTE: this decorator func call returns a decorator
# TODO: add opts: '--multicore', '--interative',
#       '--load-opts', '--save-opts', '--verbose' # TODO: use count opt for -v?
def get_options(db_df, analyze_group, **script_opts):
    print(locals())
    pass


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


if __name__ == '__main__':
    uis = get_options.main(standalone_mode=False)
