import sys

import cli
from utils.analysis import analyze_tail
from utils.settings import Settings as Settings

ui_opts = cli.get_ui_options.main(standalone_mode=False)

if ui_opts == 0:  # for catching the --help option
    sys.exit()

s = Settings(ui_opts)
ctrl_settings = s.get_settings_object('ctrl')
data_settings = s.get_settings_object('data')

analyze_tail(ctrl_settings, data_settings)
