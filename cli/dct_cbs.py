import click
import yaml
import pandas as pd

# NOTE: import detected as unused by linter, but is reified by eval() call
from _vnargs import VnargsOption


# # Decorator

def __preprocess_special_attrs_(opt_attrs):
    """Helper for correctly translating the data types from the YAML config
    and Python; and to conveniently set some meta attributes.

    This function mutates the passed opt_attrs dict
    """

    # attrs that are special expression objects
    expr_attrs = ('type', 'callback', 'cls',)
    for attr in expr_attrs:
        # check the attr is specified in the config & its value is truthy
        if attr in opt_attrs and bool(opt_attrs[attr]):
            attr_val = opt_attrs[attr]
            if isinstance(attr_val, str):
                opt_attrs[attr] = eval(attr_val)
            elif attr == 'type' and isinstance(attr_val, list):
                # branch specific to type attrs with list vals
                opt_attrs['type'] = click.Choice(attr_val)
            else:  # TODO: revise error message
                raise TypeError(f'{attr_val} of {type(attr_val)} cannot be '
                                f'used as the value for the {attr} attribute '
                                'of click.Option objects')

    # meta attrs that can be optionally passed, to customize info from --help
    meta_help_attrs = {'show_default': True, 'metavar': None}
    for attr, dflt in meta_help_attrs.items():
        opt_attrs[attr] = opt_attrs.get(attr, dflt)


# load (from YAML), get & set (preprocess) options attributes
def _load_gset_opts_attrs():

    cfg_path = 'config/options/attributes.yaml'
    with open(cfg_path) as cfg:
        opts_attrs = yaml.load(cfg, Loader=yaml.SafeLoader)

    for opt, attrs in opts_attrs.items():
        __preprocess_special_attrs_(attrs)

    return opts_attrs


# decorator wrapping click.Option's decorator API
def attach_script_opts():
    """Attach all options specified within attributes.yaml config file
    to the decorated click.Command instance.
    """
    opts_attrs = _load_gset_opts_attrs()

    def decorator(cmd):
        for opt in reversed(opts_attrs.values()):
            param_decls = opt.pop('param_decls')
            cmd = click.option(*param_decls, **opt)(cmd)
        return cmd
    return decorator


# # Callbacks

# TODO: optimize using list.index(value)?
def _get_param_from_ctx(ctx, param_name):
    for param in ctx.command.params:
        if param.name == param_name:
            return param
    else:
        raise KeyError(f'{param_name} not found in params list of '
                       f'click.Command: {ctx.command.name}')


# callback for the db_df positional argument
def gset_db_df(ctx, param, db_file):  # gset_db_df: Get/Set DataBase DataFrame
    """Open and read the passed File as a Pandas DataFrame

    If the default attr isn't manually set in the YAML config,
    infer the defaults of these extra options related to the
    database file; namely: tickers, date_i & date_f

    NOTE: the function mutates the ctx state to add the above inferred vals
    """

    db_df = pd.read_csv(db_file, index_col='Date')  # TODO: index_col case-i?
    # TODO: attach computed objects such as {df, tickers, dates} onto ctx??

    full_dates = db_df.index  # TODO: attach to ctx_obj for later access?
    # ASK/CONFIRM: using lookback good method for inferring date_i default?
    lbv = (ctx.params.get('lookback') or
           _get_param_from_ctx(ctx, 'lookback').default)  # lbv: LookBack Value

    db_extra_opts_map = {'tickers': tuple(db_df.columns),  # TODO:filter nulls?
                         'date_i': full_dates[lbv],
                         'date_f': full_dates[-1]}

    # use inferred defaults when default attr isn't manually set in YAML config
    for opt_name, infrd_dflt in db_extra_opts_map.items():
        opt = _get_param_from_ctx(ctx, opt_name)
        if opt.default is None:
            opt.default = infrd_dflt

    # TODO: consider instead of read file & return DF, just return file handle?
    return db_df
    # FIXME: performance mighe be somewhat reduced due to this IO operation???


# small mutating utility to correctly set the metavar & help attrs of xmin_args
def _set_xmin_metavar_help_(ctx):
    xmin_opt = _get_param_from_ctx(ctx, 'xmin_args')
    xmin_rule_choices = tuple(xmin_opt.default.keys())
    xmin_opt.metavar = (f"[{'|'.join(xmin_rule_choices)}]  "
                        f"[default: {xmin_rule_choices[0]}]")
    extra_help = ('-average: 2 INTs (# days) - sliding window & lag, '
                  'default: 66, 0\n') if ctx.analyse_group else ''
    xmin_opt.help = extra_help + xmin_opt.help


# callback for -G, --group
def gset_group_opts(ctx, param, analyze_group):

    # add custom top-level flag attr to Context for convenience of others
    ctx.analyse_group = False  # NOTE spelling: analySe NOT analyZe

    if analyze_group:
        ctx.analyse_group = True

        param.hidden = True  # NOTE: when -G set, hide its help

        grp_defs_fpath = 'config/options/group_defaults.yaml'
        with open(grp_defs_fpath) as cfg:
            grp_defs = yaml.load(cfg, Loader=yaml.SafeLoader)

        for opt, dflt in grp_defs.items():
            grp_opt = _get_param_from_ctx(ctx, opt)
            grp_opt.default = dflt  # NOTE: update to group specific defaults
            if grp_opt.hidden:  # NOTE: show opts hidden in non-group mode
                grp_opt.hidden = False

    # NOTE: this hacky piggybacking only works b/c -G is an eager option
    _set_xmin_metavar_help_(ctx)

    return analyze_group  # TODO: return more useful value?


# TODO: manually upgrade v.7.1+ for features: ParameterSource & show_default
# TODO/TODO: confirm click.ParameterSource & ctx.get_parameter_source usable
# callback for the xmin_args (-x, --xmin) option
def gset_xmin_args(ctx, param, xmin_args):
    rule, *vqarg = xmin_args  # vqarg: variable quantity arg(s)
    dflts_by_rule = param.default  # use default attr encoded in YAML config

    # FIXME: a bit hacky; use ctx.get_parameter_source when available
    # TODO: the 2 diff defaults are hardcoded rn; get as 1st from dflts_by_rule
    dflt_rule = 'average' if ctx.analyse_group else 'clauset'
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


# callback for options unique to -G --group mode (currently only partition)
def confirm_group_flag_set(ctx, param, val):
    if val is not None:
        assert ctx.analyse_group,\
            (f"option '{param.name}' is only available when using "
             "group tail analysis mode; set -G to use")
    else:
        # NOTE: this error should never triger as the default value &
        #       click.Choice type constaints suppresses it
        assert not ctx.analyse_group
    return val
