#  import os
#  import re
#  from statistics import NormalDist

#  import pandas as pd

import click

#  import yaml
#  from .io import load_settings


#  def _load_defaults(anal_type):
#      opt_dir = 'config/default_options/'
#
#      with open(f'{opt_dir}/shared.yaml') as shopts:
#          _shared = yaml.load(shopts, Loader=yaml.SafeLoader)
#          common = _shared['common']
#          specific = _shared[anal_type]
#
#      with open(f'{opt_dir}/{anal_type}.yaml') as spopts:
#          unique = yaml.load(spopts, Loader=yaml.SafeLoader)
#          # NOTE: "else {}" b/c normal tail analysis has no unique options
#          unique = unique if (unique is not None) else {}
#
#      return {**common, **specific, **unique}


# CLI choice constants
#  xmin_chlist = ['clauset', 'manual', 'percentile']

# TODO: add config file input option
# TODO: add interative input function & option to save as config file


#  def _get_db_choices():
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


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


def get_tails_used(tail_selected):
    """Return tuple containing the tails selected/used
    """

    use_right = True if tail_selected in ['right', 'both'] else False
    use_left = True if tail_selected in ['left', 'both'] else False

    tails_used = []
    if use_right:
        tails_used.append("right")
    if use_left:
        tails_used.append("left")

    return use_right, use_left, tuple(tails_used)


#  tickers = ["DE 01Y", "DE 03Y", "DE 05Y", "DE 10Y"]


#  @click.group(name='tail-analysis', invoke_without_command=True)
@click.command()
@click.argument('dbfile', metavar='DB_FILE', nargs=1,
                type=click.File(mode='r'),
                default='dbMSTR_test.csv')  # TODO: use callback for default?
#  help=f'select database to use: {_get_db_choices()}; or your own')
# FIXME: how to manually specify a list of tickers on the CLI?
#  @click.option('--tickers', default=["DE 01Y"], type=list)  # TODO:use config
#  @click.option('--tickers', default=tickers, type=list)  # TODO: use config
#  @click.option('--init-date', 'date_i', default='31-03-16')
#  @click.option('--final-date', 'date_f', default='5/5/2016')
# TODO: convert the 3 options above into arguments?
# TODO: in the 3 options above, autodetect tickers & dates from passed database
@click.option('-G', '--group', 'analyze_group',
              default=False, is_flag=True, is_eager=True,
              help='set this flag to use group tail analysis')
@click.option('--lookback', type=int,
              help='lookback (in days) to use for rolling analysis')
@click.option('--return-type',
              type=click.Choice(['basic', 'relative', 'log']),
              help='specify which type of series to study')
# TODO: provide as (str) choices?; rename to delta??
@click.option('--tau', type=int,
              help='specify time (in days) lag of input series')
# TODO: need to specify std/abs targets if approach is static
@click.option('--standardize/--no-standardize',
              help='normalize each investigated time series')
@click.option('--absolutize/--no-absolutize',
              help='take the absolute value of your series')
# TODO: consider using Enum types for these Choice values
@click.option('-a', '--approach',
              type=click.Choice(['static', 'rolling', 'increasing']))
# TODO: --analyze-freq is only applies to non-static approach
@click.option('--analyze-freq', 'anal_freq', type=int)
# TODO: allow specifying 'both' with '--tail left right' below (variable vals)
@click.option('-p', '--partition', hidden=True,
              help='option for group analysis')
@click.option('-t', '--tail', 'tail_selected',  # TODO: use feature switch(es)?
              type=click.Choice(['left', 'right', 'both']))
@click.option('-n', '--data-nature',
              type=click.Choice(['discrete', 'continuous']))
#  # TODO: allow None default for all xmin_rule choices (likely needs cb)
#  # TODO: likely need custom option type to allow a range of args
#  @click.option('-x', '--xmin', 'xmin_inputs',
#                # FIXME: make consistent with YAML config
#                default=('clauset', None), show_default=True,
#                type=(click.Choice(xmin_chlist), float),  # callback=xmin_cb,
#                help=f'CHOICE one of {xmin_chlist}')
#  # TODO: use callback validation for xmin_varq?
#  # TODO: and better name, ex. xmin_rule_specific_qty
#  #  @click.option('--xmin-var-qty', default=None, type=float,
#  #                help='var quantity used to calculate xmin based on rule')
@click.option('--alpha-signif', type=float,
              help='significance of the confidence interval: 1-α')
@click.option('--plpva-iter',  type=int,
              help='Clauset p-value algorithm num of iterations')
@click.option('--show-plots/--no-show-plots')
@click.option('--save-plots/--no-save-plots')
def get_options(dbfile,  # tickers, date_i, date_f,
                analyze_group, **script_opts):
    #  lookback, return_type, tau, standardize, absolutize,
    #  approach, anal_freq, partition, tail_selected, data_nature,
    #  xmin_inputs, alpha_signif, plpva_iter,
    #  show_plots, save_plots):  # TODO: add verbosity feature/flag
    # TODO: make distinction b/w private/internal & public variables
    pass

    #  if ctx.invoked_subcommand is None:
    #      defaults = _load_defaults('normal')
    #      #  uis = ctx.invoke(_get_shared_opts, **kwargs)
    #
    #  kwarg_dict = locals()

    #  script_type = 'group' if analyze_group else 'normal'
    #  sett = _load_defaults(script_type)

    #  for key, arg in kwarg_dict.items():
    #      if arg is None:
    #          kwarg_dict[key] = defaults.get(key)

    #  return kwarg_dict


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
