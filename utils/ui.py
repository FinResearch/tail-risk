import click
import yaml

#  from statistics import NormalDist
#  import pandas as pd


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


def _get_param_index(params_ls, param_name):
    for i, p in enumerate(params_ls):
        if p.name == param_name:
            return i
    else:
        return None


def set_group_opts(ctx, param, analyze_group):

    if analyze_group:

        param.hidden = True  # NOTE: when -G set, hide its help

        grp_defs_fpath = 'config/options/group_defaults.yaml'
        with open(grp_defs_fpath) as cfg:
            grp_defs = yaml.load(cfg, Loader=yaml.SafeLoader)

        for opt, dflt in grp_defs.items():
            i = _get_param_index(ctx.command.params, opt)
            grp_opt = ctx.command.params[i]
            grp_opt.default = dflt
            if grp_opt.hidden:  # NOTE: show hidden opt when in group mode
                grp_opt.hidden = False

    return analyze_group


#  def _get_db_choices():
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


#  tickers = ["DE 01Y", "DE 03Y", "DE 05Y", "DE 10Y"]

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


# TODO: add context_settings, epilog to command
@click.command(context_settings={})
@click.argument('dbfile', metavar='DB_FILE', nargs=1,
                type=click.File(mode='r'),
                default='dbMSTR_test.csv')  # TODO: use callback for default?
#  help=f'select database to use: {_get_db_choices()}; or your own')
# FIXME: how to manually specify a list of tickers on the CLI?
# TODO: convert ticker & dates options into arguments?
#  @click.option('--tickers', default=["DE 01Y"], type=list)  # TODO:use config
#  @click.option('--tickers', default=tickers, type=list)  # TODO: use config
#  @click.option('--init-date', 'date_i', default='31-03-16')
#  @click.option('--final-date', 'date_f', default='5/5/2016')
# TODO: for the 3 opts above, autodetect tickers & dates from passed db
#  #  # TODO: allow None default for all xmin_rule choices (likely needs cb)
#  #  # TODO: likely need custom option type to allow a range of args
#  #  @click.option('-x', '--xmin', 'xmin_inputs',
#  #                # FIXME: make consistent with YAML config
#  #                default=('clauset', None), show_default=True,
#  #                type=(click.Choice(xmin_chlist), float), #callback=xmin_cb,
#  #                help=f'CHOICE one of {xmin_chlist}')
#  #  # TODO: use callback validation for xmin_varq?
#  #  # TODO: and better name, ex. xmin_rule_specific_qty
#  #  #  @click.option('--xmin-var-qty', default=None, type=float,
#  #  #                help='var quantity used to calc xmin based on rule')
@click.option('-G', '--group', 'analyze_group',
              is_flag=True, is_eager=True, hidden=False,
              default=False, callback=set_group_opts,
              help=('set flag to run group analysis mode; use '
                    'with --help to also see group specifics'))
@attach_script_opts()  # NOTE: this decorator func call returns a decorator
# TODO: add opts: '--multicore', '--interative',
#       '--load-opts', '--save-opts', '--verbose'
def get_options(dbfile,  # tickers, date_i, date_f,
                analyze_group, **script_opts):
    # TODO: make distinction b/w private/internal & public variables
    print(locals())
    pass


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
