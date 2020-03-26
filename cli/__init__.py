import click
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TODO: add ROOT_DIR to sys.path in entrypoint/top-level when packaged

from .dct_cbs import gset_full_dbdf, attach_yaml_opts


@click.command(context_settings=dict(default_map=None,
                                     max_content_width=100,  # TODO: use 120?
                                     help_option_names=('-h', '--help'),
                                     #  token_normalize_func=None,
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
# TODO: support ARGs & move DB_FILE into config/options/attributes.yaml ??
@attach_yaml_opts()  # NOTE: this decorator func call returns a decorator
# FIXME: allow passing in negative number as args to opts;
#        currently interpreted as an option;
#        see: https://github.com/pallets/click/issues/555
# TODO: add opts: '--multicore', '--interative', '--gui' (using easygui),
#                 '--partial-saves', '--load-opts', '--save-opts',
#                 '--verbose' # TODO: use count opt for -v?
def get_ui_options(full_dbdf, **yaml_opts):
    # TODO: consider removing kv-pairs w/ None vals (ex. partition when no -G)?
    return dict(full_dbdf=full_dbdf, **yaml_opts)
# TODO add subcommands: plot & resume_calculation (given full/partial data),
#                       calibrate multiprocessing chunksize, etc.


# NOTE: cannot run as __main__ in packaged mode --> remove??
if __name__ == '__main__':
    ui_opts = get_ui_options.main(standalone_mode=False)

# TODO: log to stdin analysis progress (ex. 7/89 dates for ticker XYZ analyzed)
