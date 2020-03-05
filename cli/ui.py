import click
import yaml

import pandas as pd
#  from statistics import NormalDist


def _load_opts_attrs():

    attr_fpath = 'config/options/attributes.yaml'
    with open(attr_fpath) as cfg:
        opts_config = yaml.load(cfg, Loader=yaml.SafeLoader)

    opts_attrs = {}
    for opt, attrs in opts_config.items():
        opt_type = attrs.get('type')
        if opt_type is None:  # NOTE: don't explicitly set type for bool flags
            pass
        elif isinstance(opt_type, str):
            attrs['type'] = eval(opt_type)
        elif isinstance(opt_type, list):
            attrs['type'] = click.Choice(opt_type)
        else:  # TODO: revise error message
            raise TypeError(f'{opt_type} of {type(opt_type)} cannot '
                            'be used as type for click.Option')
        opts_attrs[opt] = attrs

    return opts_attrs


def attach_script_opts():
    """Attach all options specified within attributes.yaml config file
    to the decorated click.Command instance.
    """
    opts_attrs = _load_opts_attrs()

    def decorator(cmd):
        for opt in reversed(opts_attrs.values()):
            param_decls = opt.pop('param_decls')
            cmd = click.option(*param_decls, show_default=True, **opt)(cmd)
        return cmd
    return decorator


# TODO: optimize using list.index(value)?
def _get_param_from_ctx(ctx, param_name):
    for param in ctx.command.params:
        if param.name == param_name:
            return param
    else:
        raise KeyError(f'{param_name} not found in params list of '
                       f'click.Command: {ctx.command.name}')


def set_group_opts(ctx, param, analyze_group):

    if analyze_group:

        param.hidden = True  # NOTE: when -G set, hide its help

        grp_defs_fpath = 'config/options/group_defaults.yaml'
        with open(grp_defs_fpath) as cfg:
            grp_defs = yaml.load(cfg, Loader=yaml.SafeLoader)

        for opt, dflt in grp_defs.items():
            grp_opt = _get_param_from_ctx(ctx, opt)
            grp_opt.default = dflt  # NOTE: update to group specific defaults
            if grp_opt.hidden:  # NOTE: show opts hidden in non-group mode
                grp_opt.hidden = False

    return analyze_group  # TODO: return more useful value?


def _infer_db_dflt_if_unset_(ctx, param_name, inferred_val):
    """helper func used by db-related options to infer and set their
    defaults, if they're not manually specified in the YAML config

    mutates ctx state, and has no return value (as indicated by trailing _)
    """
    param = _get_param_from_ctx(ctx, param_name)
    if param.default is None:
        param.default = inferred_val


# TODO: attach computed/processed objects, full: (df, tickers, dates) onto ctx?
def gset_db_df(ctx, param, db_file):  # gset_db_df: Get/Set DataBase DataFrame

    db_df = pd.read_csv(db_file, index_col='Date')  # TODO: index_col case-i?

    # infer default tickers labels (when default not manually set in YAML cfg)
    # TODO: filter out ticker columns with null values?
    _infer_db_dflt_if_unset_(ctx, 'tickers', list(db_df.columns))
    # FIXME: need to properly parse passed list into tickers elements

    full_dates = db_df.index  # TODO: attach to ctx_obj for later access?
    # CONFIRM: lookback good method for defaulting date_i
    lbv = (ctx.params.get('lookback') or
           _get_param_from_ctx(ctx, 'lookback').default)  # lbv: LookBack Value
    # infer defaults for date_i & date_f from lkb & full_dates (when not set)
    _infer_db_dflt_if_unset_(ctx, 'date_i', full_dates[lbv])
    _infer_db_dflt_if_unset_(ctx, 'date_f', full_dates[-1])

    # TODO: consider instead of read file & return DF, just return file handle?
    return db_df
    # FIXME: performance seems to be somewhat reduced due to this IO operation


#  def _get_db_choices():
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


# CLI choice constants
#  xmin_chlist = ['clauset', 'manual', 'percentile']


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


# TODO: create VarNargsOption --> Option allowing variable numbers of args
#       also allow: i) an optional separator, examples: one of ",/ |\" etc.
#                   ii) passing the var number of args as Python list literal
# TODO: use for: --tickers, --dates, --approach, --tail, --xmin


# TODO: update conda/conda-forge channel to Click 7.1 for show_default kwarg
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
#   - remove cluttering & useless type annotations (set options' metavar attr)
@click.argument('db_df', metavar='DB_FILE', nargs=1, is_eager=True,
                type=click.File(mode='r'), callback=gset_db_df,
                default='dbMSTR_test.csv')
#  # TODO: allow None default for all xmin_rule choices (likely needs cb?)
#  # TODO: likely need custom option type to allow a range of args
#  @click.option('-x', '--xmin', 'xmin_inputs',
#                # FIXME: make consistent with YAML config
#                default=('clauset', None), show_default=True,
#                type=(click.Choice(xmin_chlist), float), #callback=xmin_cb,
#                help=f'CHOICE one of {xmin_chlist}')
#  # TODO: use callback validation for xmin_varq?
#  # TODO: and better name, ex. xmin_rule_specific_qty
#  #  @click.option('--xmin-var-qty', default=None, type=float,
#  #                help='var quantity used to calc xmin based on rule')
@click.option('-G', '--group/--no-group', 'analyze_group',
              is_eager=True, callback=set_group_opts,
              default=False, show_default=True,
              help=('set flag to run group analysis mode; use with '
                    '--help to display group options specifics'))
@attach_script_opts()  # NOTE: this decorator func call returns a decorator
# TODO: add opts: '--multicore', '--interative',
#       '--load-opts', '--save-opts', '--verbose' # TODO: use count opt for -v?
def get_options(db_df,
                analyze_group, **script_opts):
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
