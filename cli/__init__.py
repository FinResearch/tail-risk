import click
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .dct_cbs import attach_script_opts, gset_db_df, gset_group_opts


#  def _get_db_choices():
#      db_pat = re.compile(r'db.*\.(csv|xlsx)')  # TODO: confirm db name schema
#      file_matches = [db_pat.match(f) for f in os.listdir()]
#      return ', '.join([m.group() for m in file_matches if m is not None])


# TODO/TODO: update conda channel(s) to Click 7.1 for show_default kwarg
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
#   - option help texts when multiline, doesn't wrap properly
#   - remove cluttering & useless type annotations (set options' metavar attr)
@click.argument('db_df', metavar='DB_FILE', nargs=1, is_eager=True,
                type=click.File(mode='r'), callback=gset_db_df,
                default=f'{ROOT_DIR}/dbMSTR_test.csv')  # TODO: allow 2nd pos-arg for cfg?
# TODO: consider moving DB_FILE & -G into config/options/attributes.yaml ??
@click.option('-G', '--group/--no-group', 'analyze_group',
              is_eager=True, callback=gset_group_opts,
              default=False, show_default=True,
              help=('set flag to run in group analysis mode; use with --help '
                    'to also see options specific to group analysis'))
@attach_script_opts()  # NOTE: this decorator func call returns a decorator
# TODO: add opts: '--multicore', '--interative',
#       '--load-opts', '--save-opts', '--verbose' # TODO: use count opt for -v?
def get_options(db_df, analyze_group, **script_opts):
    print(locals())
    pass




# NOTE: cannot run as __main__ in packaged mode
if __name__ == '__main__':
    uis = get_options.main(standalone_mode=False)
