import click
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TODO: add ROOT_DIR to sys.path in entrypoint/top-level when packaged

from .dct_cbs import attach_yaml_opts, gset_full_dbdf, gset_group_opts


# TODO: present possible DB_FILE options if not passed & no defaults set
#  def _get_db_choices():
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


# TODO/TODO: update conda channel(s) to Click 7.1 for show_default kwarg
@click.command(context_settings=dict(default_map=None,
                                     max_content_width=100,  # TODO: use 120?
                                     help_option_names=('-h', '--help'),
                                     #  token_normalize_func=None,
                                     #  show_default=True,
                                     ),
               epilog='')
# TODO: customize --help
#   - widen first help column of options/args --> HelpFormatter.write_dl()
#   - better formatting/line wrapping for options of type click.Choice
#   - option help texts when multiline, doesn't wrap properly
#   - remove cluttering & useless type annotations (set options' metavar attr)
@click.argument('full_dbdf', metavar='DB_FILE', nargs=1, is_eager=True,
                type=click.File(mode='r'), callback=gset_full_dbdf,
                default=f'{ROOT_DIR}/dbMSTR_test.csv')
# TODO: accept optional 2nd positional argument as options config file?
@click.option('-G', '--group/--individual', 'analyze_group',
              is_eager=True, callback=gset_group_opts,
              default=False, show_default=True,
              help=('set flag to run in group analysis mode; use with --help '
                    'to also see options specific to group analysis'))
# TODO: consider moving DB_FILE & -G into config/options/attributes.yaml ??
@attach_yaml_opts()  # NOTE: this decorator func call returns a decorator
# FIXME: allow passing in negative number as args to opts;
#        currently interpreted as an option;
#        see: https://github.com/pallets/click/issues/555
# TODO: add opts: '--multicore', '--interative', '--gui' (using easygui),
#                 '--partial-saves', '--load-opts', '--save-opts',
#                 '--verbose' # TODO: use count opt for -v?
def get_ui_options(full_dbdf, analyze_group, **yaml_opts):
    return dict(full_dbdf=full_dbdf,
                analyze_group=analyze_group,
                **yaml_opts)
# TODO: consider removing kv-pairs w/ None vals (ex. partition when no -G)?
# TODO: add subcommands: plot & resume_calculation (given full/partial data)


# NOTE: cannot run as __main__ in packaged mode --> remove??
if __name__ == '__main__':
    ui_opts = get_ui_options.main(standalone_mode=False)

# TODO: log to stdin the progress (ex. 10/100 dates for ticker XYZ analyzed)
