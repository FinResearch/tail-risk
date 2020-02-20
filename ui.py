import os
import re
import pandas as pd

import click

# CLI choice constants
xmin_chlist = ['clauset', 'manual', 'percentile']

# TODO: add config file input option
# TODO: add interative input function & option to save as config file


def _get_db_choices():
    db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
    file_matches = [db_pat.match(f) for f in os.listdir()]
    return ', '.join([m.group() for m in file_matches if m is not None])


def xmin_cb(ctx, param, xmin):
    print('fired xmin callback')
    if len(xmin) == 2:
        return
    elif len(xmin) == 1:
        #  if xmin[0] == 'clauset':
        #      return ('clauset', None)
        #  elif xmin[0] == 'manual':
        #      return ('manual', 0)
        #  elif xmin[0] == 'percentile':
        #      return ('percentile', 90)
        xmin_defaults = {'clauset': None, 'manual': 0, 'percentile': 90}
        return xmin, xmin_defaults[xmin]


@click.command()
# TODO: change default to last specified
@click.option('--db-file', default='dbMSTR_test.csv',
              type=click.File(mode='r'),
              help=f'select database to use: {_get_db_choices()}; or your own')
@click.option('--tickers', default=["DE 01Y"], type=list)  # TODO:use config
@click.option('--init-date', 'date_i', default='31-03-16')
@click.option('--final-date', 'date_f', default='5/5/2016')
# TODO: in above 3 options, autodetect tickers & dates from passed database
@click.option('--lookback', default=504, show_default=True)
@click.option('--return-type', default='log', show_default=True,
              type=click.Choice(['basic', 'relative', 'log']),
              help='specify which type of series to study')
# TODO: provide as (str) choices? rename to delta??
@click.option('--tau', default=1, show_default=True,
              help='specify the time lag of the input series')
@click.option('--standardize/--no-standardize',
              default=False, show_default=True,
              help='normalize each investigated time series')
@click.option('--absolutize/--no-absolutize',
              default=False, show_default=True,
              help='take the absolute value of your series')
# TODO: consider using Enum types for these Choice values
@click.option('-a', '--approach', default='rolling', show_default=True,
              type=click.Choice(['static', 'rolling', 'increasing']))
# TODO: --analyze-freq is only applies to non-static approach
@click.option('--analyze-freq', 'anal_freq', default=1, show_default=True)
# TODO: allow specifying 'both' with '--tail left right' below (variable vals)
@click.option('-t', '--tail', 'tail_selected',
              default='both', show_default=True,
              type=click.Choice(['left', 'right', 'both']))
@click.option('-n', '--data-nature', default='continuous', show_default=True,
              type=click.Choice(['discrete', 'continuous']))
# TODO: allow None default for all xmin_rule choices (likely needs cb)
# TODO: likely need custom option type to allow a range of args
@click.option('-x', '--xmin', 'xmin_inputs',
              default=('clauset', None), show_default=True,
              type=(click.Choice(xmin_chlist), float),  # callback=xmin_cb,
              help=f'CHOICE one of {xmin_chlist}')
# TODO: use callback validation for xmin_varq?
# TODO: and better name, ex. xmin_rule_specific_qty
#  @click.option('--xmin-var-qty', default=None, type=float,
#                help='variable quantity used to calculate xmin based on rule')
@click.option('--alpha-signif', 'alpha_sgnf', default=0.05, show_default=True,
              help='significance of the confidence interval: 1-α')
@click.option('--plpva-iter', default=100, show_default=True,
              help='number of iterations for the Clauset p-value algorithm')
@click.option('--show-plots/--no-show-plots', default=True, show_default=True)
@click.option('--save-plots/--no-save-plots', default=False, show_default=True)
def get_uis(db_file, tickers, date_i, date_f, lookback, return_type, tau,
            standardize, absolutize, approach, anal_freq, tail_selected,
            data_nature, xmin_inputs, alpha_sgnf, plpva_iter,
            show_plots, save_plots):

    db_df = pd.read_csv(db_file, index_col='Date')[tickers]
    db_dates = db_df.index
    ind_i, ind_f = db_dates.get_loc(date_i), db_dates.get_loc(date_f)
    n_vec = ind_f - ind_i + 1  # FIXME: should be length of spec_dates?
    dates = db_dates[ind_i: ind_f + 1]

    labelstep = (22 if n_vec <= 252 else
                 66 if (n_vec > 252 and n_vec <= 756) else
                 121)

    # TODO: remove need for "spec_" variables
    if anal_freq > 1:
        spec_dates = dates[::anal_freq]
        spec_labelstep = 22
    elif anal_freq == 1:
        spec_dates = dates
        spec_labelstep = labelstep
    n_spdt = len(spec_dates)

    ticker_df = db_df.iloc[ind_i: ind_f + 1]

    use_right_tail = True if tail_selected in ['right', 'both'] else False
    use_left_tail = True if tail_selected in ['left', 'both'] else False
    tail_mult = 0.5 if tail_selected == 'both' else 1

    #  if xmin_rule != 'clauset' and xmin_var_qty is None:
    #      xmin_var_qty = prompt_xmin_var_qty(xmin_rule)

    #  click.echo(locals())

    return locals()


#  @click.option('--xmin-varq',  # TODO: need callback validation
#                prompt='enter variable quantity',
#                help='variable quantity used to calculate xmin based on rule')
#  def xmin_varg_prompt(ctx, param, xmin_varq):
#  def prompt_xmin_var_qty(xmin_rule):
#      var_qty_prompts = {
#          'manual': 'What is the value for xmin?',
#          'percentile': 'What is the value of the significance for xmin?'
#      }
#      default_val = {'manual': 0, 'percentile': 90}[xmin_rule]
#
#      return click.prompt(var_qty_prompts[xmin_rule], default=default_val)


#  if __name__ == '__main__':
#      get_uis()
