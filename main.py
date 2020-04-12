import sys

import cli
from utils.analysis import analyze_tail
from utils.settings import Settings as Settings

ui_opts = cli.get_user_input.main(standalone_mode=False)

if ui_opts == 0:  # for catching the --help option
    sys.exit()

#  print(ui_opts)

#  settings = Settings(ui_opts)
settings = Settings(ui_opts).settings
#  print(settings)

analyze_tail(settings)

# TODO: apply black styling to all modules (i.e. ' --> ")
# TODO: move all TODO notes into single markdown/textfile
# TODO: annotate/de-annotate NOTE notes
# TODO: move imports needed conditionally to within those branches,
#       example: like warnings.warn & itertools.product in settings.py
