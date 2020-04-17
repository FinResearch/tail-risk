# TODO: wrap logical components into classes??? (ex. group tail analysis)
# TODO: consider creating func to add Click runtime ctx obj as attr to passed
# ui_opts obj. Ex-usecase: allow settings.py access to custom dflts on VNargOpt

import click

import yaml
import pandas as pd
import os
import warnings

from pathlib import Path

# NOTE: import below is reified by eval() call, NOT unused as implied by linter
from ._vnargs import VnargsOption
from . import ROOT_DIR
OPT_CFG_DIR = f'{ROOT_DIR}/config/options/'  # TODO: use pathlib.Path ??
# TODO: once ROOT_DIR added to sys.path in project top level, ref from ROOT_DIR


# # Decorator (& its helpers) # #

def __eval_special_attrs_(opt_attrs):
    """Helper for correctly translating the data types from the YAML config
    and Python; and to conveniently set some meta attributes.

    This function mutates the passed opt_attrs dict
    """
    # attrs that need to be eval()'d from str
    expr_attrs = ('cls', 'type', 'callback',)
    for attr in expr_attrs:
        # check attr needing special treatment is spec'd for opt in YAML config
        if attr in opt_attrs:
            aval = opt_attrs[attr]  # attribute value
            if bool(aval):  # ensure value is sensible (i.e. at least truthy)
                if isinstance(aval, str):
                    opt_attrs[attr] = eval(aval)
                elif attr == 'type' and isinstance(aval, list):
                    # branch specific to type attrs with list vals
                    opt_attrs['type'] = click.Choice(aval)
                else:  # TODO: revise error message
                    raise TypeError(f"'{aval}' of type {type(aval)} cannot be "
                                    f"used as value of the '{attr}' attribute "
                                    "for click.Option objects")
            else:
                raise TypeError(f"Cannot use '{aval}' as '{attr}' for option: "
                                f"{' / '.join(opt_attrs['param_decls'])}")
    # meta attrs that can be optionally passed, to customize info from --help
    meta_help_attrs = {'show_default': True, 'metavar': None}
    for attr, dflt in meta_help_attrs.items():
        opt_attrs[attr] = opt_attrs.get(attr, dflt)


# load (from YAML), get & set (preprocess) options attributes
def _load_gset_opts_attrs():
    attrs_path = OPT_CFG_DIR + 'attributes.yaml'
    with open(attrs_path, encoding='utf8') as cfg:
        opts_attrs = yaml.load(cfg, Loader=yaml.SafeLoader)
    for opt, attrs in opts_attrs.items():
        __eval_special_attrs_(attrs)
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


# # Callbacks (CBs) # #

# # # Helpers for CBs # # #

# helper for reading a string filepath representing a datafile into a Pandas DF
def _read_fname_to_df(fname):
    fpath = Path(fname)
    fext = fpath.suffix

    if fpath.is_file():
        # TODO: move below mapping into some config file???
        ext2reader_map = {'.csv': 'csv',  # TODO: add kwargs like 'sep' & 'engine'
                          '.txt': 'table',  # 'read_table' uses \t as col-delimiter by default
                          '.xls': 'excel',  # TODO: add mult-sheet support? drop .xls support, allowing 'openpyxl' usage only
                          '.xlsx': 'excel', }
        # TODO: switch parsing engine from 'xlrd' to 'openpyxl', as former will
        # be deprecated; see: https://github.com/pandas-dev/pandas/issues/28547
        if fext in ext2reader_map.keys():
            reader = getattr(pd, f'read_{ext2reader_map[fext]}')
        else:
            raise TypeError(f"Only [{', '.join(ext2reader_map.keys())}] files "
                            f"are currently supported; given: {fpath.name}")
    else:
        raise FileNotFoundError(f"Cannot find file '{fpath.resolve()}'")

    # TODO: make index_col case-insensitive? i.e. 'Date' or 'date'
    return reader(fpath, index_col='Date')  # TODO:pd.DatetimeIndex


# TODO: optimize using list.index(value)?
def _get_param_from_ctx(ctx, param_name):
    for param in ctx.command.params:
        if param.name == param_name:
            return param
    else:
        raise KeyError(f'{param_name} not found in params list of '
                       f'click.Command: {ctx.command.name}')


# func that mutates ctx to correctly set metavar & help attrs of VnargsOption's
def _set_vnargs_choice_metahelp_(ctx):
    xmin_extra_help = (('* average : enter window & lag days (ℤ⁺, ℤ)  '
                        '[defaults: (66, 0)]\n')  # TODO: don't hardcode dflt
                       if ctx._analyze_group else '')

    vnargs_choice_opts = ('approach_args', 'xmin_args',)
    for opt in vnargs_choice_opts:
        param = _get_param_from_ctx(ctx, opt)
        choices = tuple(param.default)
        param.metavar = (f"[{'|'.join(choices)}]  [default: {choices[0]}]")
        extra_help = xmin_extra_help if opt == 'xmin_args' else ''
        param.help = extra_help + param.help

    _set_approach_args_show_default(ctx)


def _set_approach_args_show_default(ctx):
    appr_args = _get_param_from_ctx(ctx, 'approach_args')
    lkbk_dflt = _get_param_from_ctx(ctx, 'lb_override').default
    appr_args.help += f'  [defaults: ({lkbk_dflt}, 1)]'  # TODO: don't hrdcode


# TODO: checkout default_map & auto_envvar_prefix for click.Context
#       as method for setting dynamic defaults
# TODO: subsume func under approach_args CB, since xmin_args no longer uses it?
# helper for VnargsOptions to dynamically set their various defaults
def _gset_vnargs_choice_default(ctx, param, inputs,
                                errmsg_name=None,
                                errmsg_extra=None):

    dflts_by_chce = param.default  # use default map encoded in YAML config
    choices = tuple(dflts_by_chce.keys())
    dfch = choices[0]  # dfch: DeFault CHoice

    # NOTE: 'inputs' when not passed by user from CLI is taken from YAML cfg,
    # which for opts whose cbs calls this helper func, is always a dict. Thus
    # its dict.keys() iterable is passed as the arg to the 'input' parameter
    chce, *vals = inputs  # vals is always a list here, even if empty

    # ensure selected choice is in the set of possible values
    if chce not in choices:
        opt_name = errmsg_name if errmsg_name else param.name
        raise ValueError(f"'{opt_name}' {errmsg_extra if errmsg_extra else ''}"
                         f"must be one of [{', '.join(choices)}]; got: {chce}")

    # NOTE: click.ParameterSource & methods not in v7.0; using HEAD (symlink)
    opt_src = ctx.get_parameter_source(param.name)
    if opt_src == 'DEFAULT':
        vals = dflts_by_chce[dfch]
    elif opt_src == 'COMMANDLINE':
        if len(vals) == 0:
            vals = dflts_by_chce[chce]
        elif len(vals) == 1:
            vals = vals[0]  # all 1-tups are unpacked to their sole element
        else:
            # TODO: if know which num-arg in tup is more likely to vary, then
            # can implement system to allow 1 num input, and default for other
            vals = tuple(vals)
            # NOTE: tuple returned from here will always have length >= 2

    return chce, vals


# helper for converting choice types (click.Choice OR custom dict choices)
# w/ numeric str vals to Python's number types (int OR float)
def _convert_str_to_num(str_val, must_be_int=False, type_errmsg=None,
                        min_allowed=None, max_allowed=None, range_errmsg=None):
    assert isinstance(str_val, str),\
        f"value to convert to number must be of type 'str', given {str_val}"
    try:
        # TODO: confirm w/ click --> dash for negative num now works?!?
        # NOTE: curr simple but ugly hack for negative num: use _N to repr -N
        sign, str_val = ((-1, str_val[1:]) if str_val.startswith('_') else
                         (1, str_val))
        # TODO/FIXME: modify click.Option to accept '-' as -ve args on the CLI
        #             see: https://github.com/pallets/click/issues/555

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
        # preferentially return INTs over FLOATs
        return sign * (int_val if val_is_integer else float_val)
    except TypeError:
        type_errmsg = (type_errmsg or
                       f"input value must be an INT, given {str_val}")
        raise TypeError(type_errmsg)
    except AssertionError:
        range_errmsg = (range_errmsg or
                        f"number must be {comp_cond}, given {str_val}")
        raise ValueError(range_errmsg)


# TODO: create & send PR implementing this type of feature below to Click??
# helper for customizing str displayed in help msg when show_default is True
def _customize_show_default_boolcond(param, boolcond, dflt_str_2tup):
    if param.show_default:
        param.show_default = False  # turn off built-in show_default
        true_dflt, false_dflt = dflt_str_2tup
        help_dflt = true_dflt if boolcond else false_dflt
        param.help += f'  [default: {help_dflt}]'


# TODO: consider shoving _customize_show_default_boolcond into wrapper below??
# wrapper that sets the show_default attr of specific boolean flag options #
def _config_show_help_default_(ctx):  # mutates the ctx object
    nproc_cfg_val = _get_param_from_ctx(ctx, 'nproc').default
    opt_bool_map = {'run_gui': ('GUI', 'CLI'),
                    'analyze_group': ('Group', 'Individual'),
                    'norm_target': ('series', 'tail'),
                    'data_is_continuous': ('continuous', 'discrete'),
                    'run_ks_test': ('run', 'skip'),
                    'compare_distros': ('compare', 'no compare'),
                    'nproc': (f'{nproc_cfg_val} (from config)',
                              f'{os.cpu_count()} (# CPUs)'),
                    #  'plot_results': (,),
                    #  'show_plots': (,),
                    #  'save_plots': (,),
                    }
    # TODO: consider move above mapping to own config
    for opt, dflt_tup in opt_bool_map.items():
        param = _get_param_from_ctx(ctx, opt)
        _customize_show_default_boolcond(param, param.default, dflt_tup)


# # # Eager Options CBs # # #

# TODO: present possible DB_FILE options if not passed & no defaults set
#  def _get_db_choices():  # will this feature require full_dbf to be eager??
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # need to cnfrm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


# callback for -G, --group
def gset_group_opts(ctx, param, analyze_group):
    ctx._analyze_group = False  # set pvt toplvl attr on ctx for convenience

    if analyze_group:
        ctx._analyze_group = True
        opt_names = [p.name for p in ctx.command.params
                     if isinstance(p, click.Option)]
        grp_defs_fpath = OPT_CFG_DIR + 'group_defaults.yaml'
        with open(grp_defs_fpath, encoding='utf8') as cfg:
            grp_dflts = yaml.load(cfg, Loader=yaml.SafeLoader)
        for opt in opt_names:
            opt_obj = _get_param_from_ctx(ctx, opt)
            if opt in grp_dflts:  # update group specific default val
                opt_obj.default = grp_dflts[opt]
            # ONLY display options specified in group_defaults.yaml
            opt_obj.hidden = False if opt in grp_dflts else True
        param.help = ('-G set; this is the specialized help for group '
                      'tail analysis')

    # piggyback off eagerness of the -G opt to dynamically set help texts
    _set_vnargs_choice_metahelp_(ctx)
    _config_show_help_default_(ctx)

    return analyze_group


# callback for the approach option
def validate_approach_args(ctx, param, approach_args):
    try:
        approach, (lookback, anal_freq) = _gset_vnargs_choice_default(
            ctx, param, approach_args, errmsg_name='approach')
    except ValueError as err:
        if err.args[0] == 'too many values to unpack (expected 2)':
            raise ValueError("must pass both 'lookback' & 'analysis-frequency'"
                             " if overriding the default for either one")
            # TODO: can maybe constrain this, if eg. anal_freq is discrete vals
        else:
            raise ValueError(err)

    if approach == 'static':
        assert lookback is None and anal_freq is None,\
            ("approach 'static' does not take 'lookback' & "
             "'analysis-frequency' arguments")
    elif (approach in {'rolling', 'increasing'} and
          all(isinstance(val, str) for val in (lookback, anal_freq))):
        type_errmsg = ("both 'lookback' & 'analysis-frequency' args for "
                       f"approach '{approach}' must be INTs (# days); "
                       f"given: {lookback}, {anal_freq}")
        lookback, anal_freq = [_convert_str_to_num(val, must_be_int=True,
                                                   type_errmsg=type_errmsg,
                                                   min_allowed=1)
                               for val in (lookback, anal_freq)]
    else:  # FIXME/TODO: this branch will never get reached, no? -> remove?
        raise TypeError(f"approach '{approach}' is incompatible with "
                        f"inputs: '{lookback}', '{anal_freq}'")

    # set as toplvl ctx attrs for the convenience of the gset_full_dbdf cb
    ctx._approach, ctx._lookback = approach, lookback
    return approach, lookback, anal_freq


# # # Ordinary CBs # # #

# callback for the full_dbdf positional Argument (NOT Option)
def gset_full_dbdf(ctx, param, db_fname):
    """Open and read the passed string filepath as a Pandas DataFrame. Then
    infer default values for {tickers, date_i & date_f} from the loaded DF,
    if they were not manually set inside of: config/options/attributes.yaml

    NOTE: the function mutates the ctx state to add the inferred default vals
    """
    # TODO: attach calc'd objs such as (df, tickers, dates) onto ctx for use??

    full_dbdf = _read_fname_to_df(db_fname)

    full_dates = full_dbdf.index
    # inferred index of date_i; only used when 'default' attr not set in YAML
    di_iix = 0 if ctx._approach == 'static' else ctx._lookback - 1

    dbdf_attrs = {'tickers': list(full_dbdf.columns),  # TODO: rm NaNs/nulls??
                  'date_i': full_dates[di_iix],
                  'date_f': full_dates[-1]}

    # use inferred defaults when default attr isn't manually set in YAML config
    for opt_name, infrd_dflt in dbdf_attrs.items():
        opt = _get_param_from_ctx(ctx, opt_name)
        if opt.default is None:
            opt.default = infrd_dflt

    # TODO: consider instead of read file & return DF, just return file handle?
    return full_dbdf
    # FIXME: performance mighe be somewhat reduced due to this IO operation???


#  def set_tickers_from_textfile(ctx, param, tickers):
#      pass


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


def determine_lookback_override(ctx, param, lb_ov):
    lb_src = ctx.get_parameter_source(param.name)
    if lb_src == 'DEFAULT' or ctx._approach == 'static':
        if lb_src == 'COMMANDLINE':
            warnings.warn("'--lb / --lookback' N/A to STATIC approach; "
                          f"ignoring passed value of {lb_ov}")
        lb_ov = None
    return lb_ov

#  # callback for the --tau option
#  def cast_tau(ctx, param, tau_str):
#      # NOTE: the must_be_int flag is unneeded since using click.Choice
#      return _convert_str_to_num(tau_str, must_be_int=True)


def validate_norm_target(ctx, param, target):
    tmap = {True: '--norm-series', False: '--norm-tail'}
    tgt_src = ctx.get_parameter_source(param.name)
    # norm_target only applies to individual tail analysis w/ static approach
    if ctx._analyze_group or ctx._approach != 'static':
        if tgt_src == 'COMMANDLINE':
            # TODO: use warnings.showwarning() to write to sys.stdout??
            warnings.warn('Normalization target is only applicable to STATIC '
                          'approach in INDIVIDUAL mode, i.e. w/ "-a static" & '
                          f'no "-G" set. Ignoring flag {tmap[target]}')
        return None
    return target


# callback for the xmin_args (-x, --xmin) option
def parse_xmin_args(ctx, param, xmin_args):
    """there are 5 types of accepted input to --xmin:
    * average:     : "$ ... -x 66 5" (only applicable in -G mode)
    * XMINS_FILE   : "$ ... -x xmins_data_file.txt"
    * clauset      : "$ ... -x clauset"
    * % (percent)  : "$ ... -x 99%"
    * ℝ (manual)   : "$ ... -x 0.5" OR "$ ... -x _2" (_ denotes negatives)
    """

    if ctx.get_parameter_source(param.name) == "DEFAULT":
        xmin_args = ('66', '0',) if ctx._analyze_group else ('clauset',)

    x, *y = xmin_args

    if bool(y):  # this can only possibly be the average method
        assert ctx._analyze_group and ctx._approach != 'static',\
            (f"multiple args {xmin_args} passed to '-x / --xmin', thus use "
             "method 'average', which is only applicable w/ a dynamic "
             "approach & in group analysis mode (i.e. -G flag set)")
        if len(y) == 2:
            a, b, c = xmin_args
            if all(s.isdecimal() for s in (a, b)):
                win, lag, fname = xmin_args
            elif all(s.isdecimal() for s in (b, c)):
                fname, win, lag = xmin_args
            else:
                errmsg = ("3 args passed to '--xmin'; xmins data file to use "
                          "for 'average' must be passed either FIRST or LAST")
                raise AssertionError(errmsg)
        elif len(y) == 1:
            win, lag, fname = (*xmin_args, None)
        # TODO: account for when only 1 num arg passed --> make it window-size?
        type_errmsg = ("both numeric args to '--xmin' rule 'average' must "
                       f"be INTs (# days); given: '{win}, {lag}'")
        win, lag = sorted([_convert_str_to_num(val, must_be_int=True,
                                               type_errmsg=type_errmsg,
                                               min_allowed=0)
                           # TODO: enable diff min for window & lag args
                           for val in (win, lag)], reverse=True)
        ctx._xmins_df = _read_fname_to_df(fname) if bool(fname) else None
        return ('average', (win, lag, ctx._xmins_df))

    try:  # if try successful, necessarily must be the XMIN_FILE
        ctx._xmins_df = _read_fname_to_df(x)
        return ('file', ctx._xmins_df)
    except FileNotFoundError:
        pass

    if x == 'clauset':
        return ('clauset', None)
    elif x.endswith('%'):
        # ASK/TODO: use '<=' OR is '<' is okay?? i.e. open or closed bounds
        range_errmsg = f"percent value must be in [0, 100]; given: {x[:-1]}"
        percent = _convert_str_to_num(x[:-1], min_allowed=0, max_allowed=100,
                                      range_errmsg=range_errmsg)
        return ('percent', percent)
    else:
        try:
            return ('manual', _convert_str_to_num(x))
        except ValueError:
            raise ValueError(f"option '-x / --xmin' is incompatible w/ input: "
                             f"'{x}'; see --help for acceptable inputs")


def gset_nproc_default(ctx, param, nproc):
    return nproc or os.cpu_count()


# # Post-Parsing Functions # #
# # for opts requiring full completed ctx AND/OR
# # actions requiring parse-order independence
# # also note that they mutate yaml_opts (denoted by _-suffix)

# called in conditionalize_normalization_options_ below
def __validate_norm_timings_(ctx, yaml_opts):
    smap = {opt: ctx.get_parameter_source(opt) for
            opt in ('norm_before', 'norm_after')}
    if not ctx._analyze_group:
        for opt, src in smap.items():
            if src == 'COMMANDLINE':
                warnings.warn('Normalization timing only applicable in GROUP '
                              'analysis mode, i.e. w/ the "-G" flag set. '
                              f"Ignoring flag --{'-'.join(opt.split('_'))}")
            yaml_opts[opt] = None
    return smap  # return mapping of norm-timing opts sources for convenience


def conditionalize_normalization_options_(ctx, yaml_opts):
    timing_srcs = __validate_norm_timings_(ctx, yaml_opts)
    use_default_timing = all(src == 'DEFAULT' for src in timing_srcs.values())
    norm_srcs = {**timing_srcs,  # TODO: make top-level ctx attr for all srcs?
                 'norm_target': ctx.get_parameter_source('norm_target')}

    normalize = yaml_opts['standardize'] or yaml_opts['absolutize']
    if not normalize:
        # the default case of no normalization at all
        for opt in ('norm_target', 'norm_before', 'norm_after'):
            if yaml_opts[opt] is not None and norm_srcs[opt] != "DEFAULT":
                warnings.warn(f"Norm option '{opt}' only applicable w/ --std "
                              f"and/or --abs set; ignoring option {opt}")
                yaml_opts[opt] = None
    elif normalize and ctx._analyze_group and use_default_timing:
        # set default norm timings if none explicitly set (but --std/--abs set)
        yaml_opts['norm_before'] = False
        yaml_opts['norm_after'] = True
        # TODO: just set the defaults in YAML config?


# validate then correctly set/toggle the two tail selection options
def conditionally_toggle_tail_flag_(ctx, yaml_opts):
    # NOTE: this function is agnostic of which tail name is passed first
    names_srcs_vals = [(t, ctx.get_parameter_source(t), yaml_opts[t])
                       for t in ('anal_right', 'anal_left')]
    names, sources, values = zip(*names_srcs_vals)

    if not any(values):
        if all(src == 'DEFAULT' for src in sources):
            raise ValueError('defaults for both tails are False (skip); '
                             'specify -L or -R to analyze the left/right tail;'
                             ' or -LR for both')
        raise ValueError('at least one tail must be selected to run analysis')

    # only toggle one of the tail selection when they are specified via diffent
    # sources (i.e. 1 COMMANDLINE, 1 DEFAULT), and they are both True
    if sources[0] != sources[1] and all(values):
        for name, src, val in names_srcs_vals:
            yaml_opts[name] = val if src == 'COMMANDLINE' else not val

    if yaml_opts['absolutize']:
        if ctx.get_parameter_source('anal_left') == 'COMMANDLINE':
            warnings.warn("'--abs / --absolutize' flag set, only RIGHT tail "
                          "appropriate for analysis; ignoring -L, using -R")
        yaml_opts['anal_left'] = False
        yaml_opts['anal_right'] = True


# helper for validating analysis dates
def _assert_dates_in_df(df, dates_to_check):
    missing_dates = [dt for dt in dates_to_check if dt not in df.index]
    if bool(missing_dates):
        raise ValueError(f"analysis date(s) {missing_dates} needed but NOT "
                         f"found in Date Index of loaded DataFrame:\n\n{df}\n")


def validate_df_date_indexes(ctx, yaml_opts):
    di, df = yaml_opts['date_i'], yaml_opts['date_f']
    full_dbdf = ctx.params['full_dbdf']
    _assert_dates_in_df(full_dbdf, (di, df))

    if hasattr(ctx, '_xmins_df') and isinstance(ctx._xmins_df, pd.DataFrame):
        anal_freq = yaml_opts['approach_args'][2]
        anal_dates = full_dbdf.loc[di:df:anal_freq].index
        _assert_dates_in_df(ctx._xmins_df, anal_dates)


post_proc_funcs = (conditionalize_normalization_options_,
                   conditionally_toggle_tail_flag_,
                   validate_df_date_indexes,)
