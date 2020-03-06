import yaml
import pandas as pd


# TODO: optimize using list.index(value)?
def _get_param_from_ctx(ctx, param_name):
    for param in ctx.command.params:
        if param.name == param_name:
            return param
    else:
        raise KeyError(f'{param_name} not found in params list of '
                       f'click.Command: {ctx.command.name}')


# callback for the db_df positional argument
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


# helper for gset_db_df
def _infer_db_dflt_if_unset_(ctx, param_name, inferred_val):
    """helper func used by db-related options to infer and set their
    defaults, if they're not manually specified in the YAML config

    mutates ctx state, and has no return value (as indicated by trailing _)
    """
    param = _get_param_from_ctx(ctx, param_name)
    if param.default is None:
        param.default = inferred_val


# callback for -G, --group
def gset_group_opts(ctx, param, analyze_group):

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


# TODO: manually upgrade v.7.1+ for features: ParameterSource & show_default
# TODO/TODO: confirm click.ParameterSource & ctx.get_parameter_source usable
def gset_xmin_args(ctx, param, xmin_args):
    rule, *vqarg = xmin_args  # vqarg: variable quantity arg(s)
    dflts_by_rule = param.default  # use default attr encoded in YAML config

    # FIXME: a bit hacky; use ctx.get_parameter_source when available
    # TODO: the 2 diff defaults are hardcoded rn; get as 1st from dflts_by_rule
    dflt_rule = 'average' if ctx.params['analyze_group'] else 'clauset'
    if rule == dflt_rule:
        vqarg = dflts_by_rule[dflt_rule]

    # ensure selected xmin_rule is a valid choice
    if rule not in dflts_by_rule:
        raise ValueError('xmin determination rule must be one of: '
                         f"{', '.join(dflts_by_rule.keys())}")

    if not bool(vqarg):  # True if 'vqarg is None' OR 'len(vqarg) == 0'
        xmin_args = rule, dflts_by_rule[rule]
    elif len(vqarg) == 1:
        xmin_args = rule, vqarg[0]
    elif rule == 'average' and len(vqarg) == 2:  # NOTE: only applies w/ -G
        # ASK/TODO: window > lag (i.e. sliding window vs. lag, in days) always
        window, lag = sorted(vqarg, reverse=True)
        xmin_args = rule, window, lag
    else:
        raise TypeError(f"xmin determination by '{rule}' rule is incompatible "
                        f"with inputs: {', '.join(vqarg)}")

    return xmin_args  # NOTE: the number args might be in string form


