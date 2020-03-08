import click
import yaml
import pandas as pd

from ._vnargs import VnargsOption  # NOTE: linter detect as unused; reified by eval() call
from . import ROOT_DIR
OPT_CFG_DIR = f'{ROOT_DIR}/config/options/'  # TODO: use pathlib.Path ??


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

    attrs_path = OPT_CFG_DIR + 'attributes.yaml'
    with open(attrs_path) as cfg:
        opts_attrs = yaml.load(cfg, Loader=yaml.SafeLoader)

    for opt, attrs in opts_attrs.items():
        __preprocess_special_attrs_(attrs)

    return opts_attrs


# decorator wrapping click.Option's decorator API
def attach_yaml_opts():
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
# gset_full_dbdf: Get/Set Full DataBase DataFrame
def gset_full_dbdf(ctx, param, db_file):
    """Open and read the passed File as a Pandas DataFrame

    If the default attr isn't manually set in the YAML config,
    infer the defaults of these extra options related to the
    database file; namely: tickers, date_i & date_f

    NOTE: the function mutates the ctx state to add the above inferred vals
    """

    db_df = pd.read_csv(db_file, index_col='Date')  # TODO: index_col case-i?
    full_dbdf = pd.read_csv(db_file, index_col='Date')  # TODO: index_col case-i?
    # TODO: attach computed objects such as {df, tickers, dates} onto ctx??

    full_dates = full_dbdf.index  # TODO: attach to ctx_obj for later access?
    # ASK/CONFIRM: using lookback good method for inferring date_i default?
    lbv = (ctx.params.get('lookback') or
           _get_param_from_ctx(ctx, 'lookback').default)  # lbv: LookBack Value

    db_extra_opts_map = {'tickers': list(full_dbdf.columns),  # TODO:filter nulls?
                         'date_i': full_dates[lbv],
                         'date_f': full_dates[-1]}

    # use inferred defaults when default attr isn't manually set in YAML config
    for opt_name, infrd_dflt in db_extra_opts_map.items():
        opt = _get_param_from_ctx(ctx, opt_name)
        if opt.default is None:
            opt.default = infrd_dflt

    # TODO: consider instead of read file & return DF, just return file handle?
    return full_dbdf
    # FIXME: performance mighe be somewhat reduced due to this IO operation???


# small mutating utility to correctly set the metavar & help attrs of xmin_args
def _set_vnargs_choice_metahelp_(ctx):
    xmin_extra_help = ('-average: 2 INTs (# days) as window & lag, '
                       'default: 66, 0\n') if ctx.analyse_group else ''

    vnargs_choice_opts = ('approach', 'xmin_args',)

    for opt_name in vnargs_choice_opts:
        opt_obj = _get_param_from_ctx(ctx, opt_name)
        opt_choices = tuple(opt_obj.default.keys())
        opt_obj.metavar = (f"[{'|'.join(opt_choices)}]  "
                           f"[default: {opt_choices[0]}]")
        extra_help = xmin_extra_help if opt_name == 'xmin_args' else ''
        opt_obj.help = extra_help + opt_obj.help


# callback for -G, --group
def gset_group_opts(ctx, param, analyze_group):

    # add custom top-level flag attr to Context for convenience of others
    ctx.analyse_group = False  # NOTE spelling: analySe NOT analyZe

    if analyze_group:
        ctx.analyse_group = True

        param.hidden = True  # NOTE: when -G set, hide its help

        grp_defs_fpath = OPT_CFG_DIR + 'group_defaults.yaml'
        with open(grp_defs_fpath) as cfg:
            grp_defs = yaml.load(cfg, Loader=yaml.SafeLoader)

        for opt, dflt in grp_defs.items():
            grp_opt = _get_param_from_ctx(ctx, opt)
            grp_opt.default = dflt  # NOTE: update to group specific defaults
            if grp_opt.hidden:  # NOTE: show opts hidden in non-group mode
                grp_opt.hidden = False

    # NOTE: this hacky piggybacking only works b/c -G is an eager option
    _set_vnargs_choice_metahelp_(ctx)

    return analyze_group  # TODO: return more useful value?


def _gset_vnargs_choice_default(ctx, param, inputs, dflt=None):

    dflts_by_chce = param.default  # use default map encoded in YAML config
    choices = tuple(dflts_by_chce.keys())

    chce, *vals = inputs  # NOTE: here vals is always a list

    # ensure selected choice is in the set of possible values
    if chce not in choices:
        raise ValueError(f"'{param.name}' must be one of: "
                         f"{', '.join(choices)}; given: {chce}")

    # set the default to the 1st entry of the choice list when dflt not given
    if dflt is None:
        dflt = choices[0]

    # NOTE: click.ParameterSource unavailable in v7.0; using HEAD (symlink)
    opt_src = ctx.get_parameter_source(param.name)
    if opt_src == click.ParameterSource.DEFAULT:
        vals = dflts_by_chce[dflt]
    elif opt_src == click.ParameterSource.COMMANDLINE:
        if len(vals) == 0:
            vals = dflts_by_chce[chce]
        elif len(vals) == 1:
            vals = vals[0]
        else:
            vals = tuple(vals)

    return chce, vals


# callback for the approach option
def validate_approach_args(ctx, param, approach_args):

    approach, freq = _gset_vnargs_choice_default(ctx, param, approach_args)

    if approach == 'static':
        assert freq is None,\
            "approach 'static' does not take extra args"
    elif approach in ('rolling', 'increasing') and isinstance(freq, str):
        try:
            freq_float = float(freq)
            freq_int = int(freq_float)
            assert freq_int == freq_float
            freq = freq_int
        except (ValueError, AssertionError):
            raise TypeError(f"argument to approach '{approach}' must be "
                            f"an INT (# days); given: {freq}")
    else:  # NOTE/TODO: this branch will never get reached, right? -> remove?
        raise TypeError(f"approach '{approach}' is incompatible "
                        f"with inputs: {freq}")

    return approach, freq


# callback for the xmin_args (-x, --xmin) option
def validate_xmin_args(ctx, param, xmin_args):

    dflt_rule = 'average' if ctx.analyse_group else 'clauset'
    rule, vqarg = _gset_vnargs_choice_default(ctx, param, xmin_args, dflt_rule)

    try:
        if rule == 'clauset':
            assert vqarg is None,\
                "xmin determination rule 'clauset' does not take extra args"
        elif rule in ('manual', 'percentile') and isinstance(vqarg, str):
            vqarg_float = float(vqarg)
            vqarg_int = int(vqarg_float)
            vqarg = vqarg_int if vqarg_int == vqarg_float else vqarg_float
        elif rule == 'average' and len(vqarg) == 2:  # NOTE: only applies w/ -G
            # TODO: need to account for when only given 1 value for average??
            # ASK/TODO: window > lag (sliding window vs. lag, in days) always
            vqarg_str = vqarg
            vqarg_float = tuple(map(float, vqarg))
            vqarg = tuple(map(int, vqarg_float))
            assert vqarg == vqarg_float,\
                {"both args to xmin rule 'average' must be INTs (# days); "
                 f"given: {', '.join(vqarg_str)}"}
        else:
            raise TypeError(f"xmin determination rule '{rule}' rule is "
                            f"incompatible with inputs: {vqarg}")
    except ValueError:
        raise TypeError(f"arg(s) to xmin determination rule '{rule}' must be "
                        f"number(s); given: {vqarg}")

    return rule, vqarg   # NOTE: num args are all of str type (incl. defaults)


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
