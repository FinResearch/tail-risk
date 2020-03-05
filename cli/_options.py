#  from statistics import NormalDist
#  import pandas as pd
import click
import yaml


def _get_group_opts_attrs(opts_attrs):

    group_dir = 'config/options/group/'

    with open(f'{group_dir}/defaults.yaml') as f:
        grp_defs = yaml.load(f, Loader=yaml.SafeLoader)
    for opt, val in grp_defs.items():
        opts_attrs[opt]['default'] = val

    grp_opts_attrs = _load_opts_attrs(f'{group_dir}/attributes.yaml')

    # TODO: better organize the order of the options in help text
    return {**opts_attrs, **grp_opts_attrs}


def _load_opts_attrs(fpath):

    with open(fpath) as f:
        options = yaml.load(f, Loader=yaml.SafeLoader)

    opts_attrs = {}
    for opt, attrs in options.items():
        opt_type = attrs.get('type')
        if opt_type is None:  # NOTE: no need to explicitly set type for flags
            pass
        elif isinstance(opt_type, list):
            attrs['type'] = click.Choice(opt_type)
        elif isinstance(opt_type, str):
            attrs['type'] = eval(opt_type)
        else:  # TODO: revise error message
            raise TypeError(f'{opt_type} of {type(opt_type)} cannot '
                            'be used as type for click.Option')
        opts_attrs[opt] = attrs

    return opts_attrs


# TODO: separate options definition config file from default values config file
def attach_options(ctx, param, analyse_group):

    attr_fpath = 'config/options/attributes.yaml'
    opts_attrs = _load_opts_attrs(attr_fpath)

    if analyse_group:
        opts_attrs = _get_group_opts_attrs(opts_attrs)

    for opt in opts_attrs.values():
        option = click.Option(show_default=True, **opt)
        ctx.command.params.append(option)

    return analyse_group


# TODO: once conda/conda-forge updates Click to 7.1
#       add show_default=True into context_settings
@click.command(context_settings=dict(max_content_width=100,  # TODO: use 120?
                                     help_option_names=('-h', '--help'),))
@click.argument('dbfile', metavar='DB_FILE', nargs=1,
                type=click.File(mode='r'),
                default='dbMSTR_test.csv')  # TODO: use callback for default
@click.option('-G', '--group', 'analyse_group', is_eager=True,
              is_flag=True, default=False, show_default=True,
              callback=attach_options,
              help='set flag to run group tail analysis')
# TODO: add custom --help message, informing of combination with -G option
# TODO: widen first help column of options/args --> HelpFormatter.write_dl()
# TODO: better formatting/line wrapping for options of type click.Choice
def get_options(dbfile, analyse_group, **script_opts):
    pass


def set_context(kwd):
    pass


if __name__ == '__main__':
    uis = get_options.main(standalone_mode=False)
