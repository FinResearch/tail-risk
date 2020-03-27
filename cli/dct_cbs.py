import click
from click.core import ParameterSource

import yaml
import pandas as pd
import os

# NOTE: import below is reified by eval() call, NOT unused as implied by linter
from ._vnargs import VnargsOption
from . import ROOT_DIR
OPT_CFG_DIR = f'{ROOT_DIR}/config/options/'  # TODO: use pathlib.Path ??
# TODO: once ROOT_DIR added to sys.path in project top level, ref from ROOT_DIR


# # Decorator # #

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


# # Callbacks # #

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

    # TODO: make index_col case-insensitive? i.e. 'Date' or 'date'
    full_dbdf = pd.read_csv(db_file, index_col='Date')  # TODO:pd.DatetimeIndex
    # TODO: attach computed objects such as {df, tickers, dates} onto ctx??

    full_dates = full_dbdf.index  # TODO: attach to ctx_obj for later access?
    # FIXME: determine how to infer date_i w/ lookback None under 'static' appr
    lbv = (ctx.params.get('lookback') or
           _get_param_from_ctx(ctx, 'lookback').default)  # lbv: LookBack Value

    # TODO: under static approach, use 0-th index for inferred date_i?
    db_extra_opts_map = {'tickers': list(full_dbdf.columns),  # TODO: rm nulls?
                         'date_i': full_dates[lbv-1],
                         'date_f': full_dates[-1]}

    # use inferred defaults when default attr isn't manually set in YAML config
    for opt_name, infrd_dflt in db_extra_opts_map.items():
        opt = _get_param_from_ctx(ctx, opt_name)
        if opt.default is None:
            opt.default = infrd_dflt

    # TODO: consider instead of read file & return DF, just return file handle?
    return full_dbdf
    # FIXME: performance mighe be somewhat reduced due to this IO operation???


# TODO: present possible DB_FILE options if not passed & no defaults set
#  def _get_db_choices():
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


#  def set_tickers_from_textfile(ctx, param, tickers):
#      pass


# small mutating utility to correctly set the metavar & help attrs of xmin_args
def _set_vnargs_choice_metahelp_(ctx):
    xmin_extra_help = ('-average: 2 INTs (# days) as window & lag, '
                       'default: 66, 0\n') if ctx._analyze_group else ''

    vnargs_choice_opts = ('approach_args', 'xmin_args',)

    for opt_name in vnargs_choice_opts:
        opt_obj = _get_param_from_ctx(ctx, opt_name)
        opt_choices = tuple(opt_obj.default.keys())
        opt_obj.metavar = (f"[{'|'.join(opt_choices)}]  "
                           f"[default: {opt_choices[0]}]")
        extra_help = xmin_extra_help if opt_name == 'xmin_args' else ''
        opt_obj.help = extra_help + opt_obj.help


# callback for -G, --group
def gset_group_opts(ctx, param, analyze_group):

    _customize_show_default_boolcond(param, analyze_group,
                                     ('group', 'individual'))

    # private toplevel group flag-val on Context for convenience of other cbs
    ctx._analyze_group = False  # NOTE spelling: analySe NOT analyZe

    if analyze_group:
        ctx._analyze_group = True

        opt_names = [p.name for p in ctx.command.params
                     if isinstance(p, click.Option)]
        grp_defs_fpath = OPT_CFG_DIR + 'group_defaults.yaml'
        with open(grp_defs_fpath) as cfg:
            grp_dflts = yaml.load(cfg, Loader=yaml.SafeLoader)

        for opt in opt_names:
            opt_obj = _get_param_from_ctx(ctx, opt)
            if opt in grp_dflts:  # update group specific default val
                opt_obj.default = grp_dflts[opt]
            # show opts hidden in individual mode, & hide opts common to both
            opt_obj.hidden = False if opt in grp_dflts else True

        param.hidden = False  # undoes the opt_obj.hidden toggle above
        param.help = 'see below for help specific to group analysis'

    # NOTE: this hacky piggybacking only works b/c -G is an eager option
    _set_vnargs_choice_metahelp_(ctx)

    return analyze_group  # TODO: return more useful value?


# helper for converting choice types (click.Choice OR custom dict choices)
# w/ numeric str vals to Python's number types (int OR float)
def _convert_str_to_num(str_val, must_be_int=False, type_errmsg=None,
                        min_allowed=None, max_allowed=None, range_errmsg=None):
    assert isinstance(str_val, str),\
        f"value to convert to number must be of type 'str', given {str_val}"
    try:
        float_val = float(str_val)
        int_val = int(float_val)
        val_is_integer = int_val == float_val
        if must_be_int and not val_is_integer:
            raise TypeError
        if min_allowed is not None and float_val < min_allowed:
            comp_cond = f'>= {min_allowed}'
            raise AssertionError
        if max_allowed is not None and float_val > max_allowed:
            comp_cond = f'<= {max_allowed}'
            raise AssertionError
        return int_val if val_is_integer else float_val  # prefer return INT
    except (ValueError, TypeError):
        type_errmsg = (f"input value must be an INT, given {str_val}"
                       if type_errmsg is None else type_errmsg)
        raise TypeError(type_errmsg)
    except AssertionError:
        range_errmsg = (f"number must be {comp_cond}, given {str_val}"
                        if range_errmsg is None else range_errmsg)
        raise ValueError(range_errmsg)


# callback for the --tau option
def cast_tau(ctx, param, tau_str):
    # NOTE: the must_be_int flag is unneeded since using click.Choice
    return _convert_str_to_num(tau_str, must_be_int=True)


# callback for the lookback option
def gset_lookback(ctx, param, lookback):
    approach, _ = ctx.params.get('approach_args')
    # FIXME: no lookback for static -> need new method to infer date_i
    if approach == 'static':
        return None
    return lookback


# helper for VnargsOption's
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
    if opt_src == ParameterSource.DEFAULT:
        vals = dflts_by_chce[dflt]
    elif opt_src == ParameterSource.COMMANDLINE:
        if len(vals) == 0:
            vals = dflts_by_chce[chce]
        elif len(vals) == 1:
            vals = vals[0]
        else:
            vals = tuple(vals)

    return chce, vals


# callback for the approach option
def validate_approach_args(ctx, param, approach_args):

    approach, anal_freq = _gset_vnargs_choice_default(ctx, param,
                                                      approach_args)

    if approach == 'static':
        assert anal_freq is None,\
            "approach 'static' does not take extra args"
    elif approach in {'rolling', 'increasing'} and isinstance(anal_freq, str):
        type_errmsg = (f"anal_frequency arg to approach '{approach}' must be "
                       f"an INT (# days); given: {anal_freq}")
        anal_freq = _convert_str_to_num(anal_freq, min_allowed=1,
                                        must_be_int=True,
                                        type_errmsg=type_errmsg)
    else:  # NOTE/TODO: this branch will never get reached, right? -> remove?
        raise TypeError(f"approach '{approach}' is incompatible "
                        f"with inputs: {anal_freq}")

    return approach, anal_freq


# callback for the xmin_args (-x, --xmin) option
def validate_xmin_args(ctx, param, xmin_args):

    dflt_rule = 'average' if ctx._analyze_group else 'clauset'
    rule, vqarg = _gset_vnargs_choice_default(ctx, param, xmin_args, dflt_rule)

    try:
        if rule == 'clauset':  # TODO: move 'clauset' out of 'try' block?
            assert vqarg is None,\
                "xmin determination rule 'clauset' does not take extra args"
        elif rule == 'manual' and isinstance(vqarg, str):
            # FIXME: need to allow negative number inputs on CLI for 'manual'
            vqarg = _convert_str_to_num(vqarg)
        elif rule == 'percentile' and isinstance(vqarg, str):
            range_errmsg = ("xmin determination rule 'percentile' takes "
                            "a number between 0 and 100")
            # ASK/TODO: use '<=' OR is '<' is okay??
            vqarg = _convert_str_to_num(vqarg, min_allowed=0, max_allowed=100,
                                        range_errmsg=range_errmsg)
        elif rule == 'average' and len(vqarg) == 2:  # NOTE: only applies w/ -G
            # TODO: need to account for when only given 1 value for average??
            type_errmsg = ("both args to xmin rule 'average' must be "
                           f"INTs (# days); given: {', '.join(vqarg)}")
            vqarg = [_convert_str_to_num(val, must_be_int=True,
                                         type_errmsg=type_errmsg,
                                         min_allowed=0)
                     # TODO: enable diff min_allowed for window & lag args
                     for val in vqarg]
            vqarg = tuple(sorted(vqarg, reverse=True))  # always: window > lag
        else:
            raise TypeError(f"xmin determination rule '{rule}' rule is "
                            f"incompatible with inputs: {vqarg}")
    except (TypeError, ValueError, AssertionError):
        raise

    return rule, vqarg   # NOTE: num args are all of str type (incl. defaults)


# callback for options unique to -G --group mode (curr. only for --partition)
def confirm_group_flag_set(ctx, param, val):
    if val is not None:
        assert ctx._analyze_group,\
            (f"option '{param.name}' is only available when using "
             "group tail analysis mode; set -G or --group to use")
    else:
        # NOTE: this error should never triger as the default value &
        #       click.Choice type constraints suppresses it
        assert not ctx._analyze_group
    return val


# helper for customizing str displayed in help msg when show_default is True
def _customize_show_default_boolcond(param, boolcond, dflt_str_2tup):
    if param.show_default:
        param.show_default = False  # turn off built-in show_default
        true_dflt, false_dflt = dflt_str_2tup
        help_dflt = true_dflt if boolcond else false_dflt
        param.help += f'  [default: {help_dflt}]'
# TODO: create and send this feature as PR to pallets/click ??


# TODO: reorder callback definitions in this file
# callback for --ks-test / --ks-skip
def customize_ksflag_show_default(ctx, param, ks_flag):
    _customize_show_default_boolcond(param, ks_flag,
                                     ('Run', 'Skip',))
    return ks_flag


def get_nproc_set_show_default(ctx, param, nproc):
    conf_dflt = _get_param_from_ctx(ctx, param.name).default
    conf_help = f'{conf_dflt} (Config File)'
    none_help = f'{os.cpu_count()} (# CPUs)'
    _customize_show_default_boolcond(param, conf_dflt, (conf_help, none_help))
    return os.cpu_count() if nproc is None else nproc
