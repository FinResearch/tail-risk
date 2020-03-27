import sys

import cli
from utils.analysis import analyze_tail
from utils.settings import Settings as Settings

ui_opts = cli.get_ui_options.main(standalone_mode=False)

if ui_opts == 0:  # for catching the --help option
    sys.exit()
else:
    print(ui_opts)

settings = Settings(ui_opts).settings
# TODO: apply black styling to all modules (i.e. ' --> ")

analyze_tail(settings)
