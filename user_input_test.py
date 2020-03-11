import cli
from utils.settings import Settings as Settings

ui_opts = cli.get_ui_options.main(standalone_mode=False)
s = Settings(ui_opts)
ctrl_settings = s.get_settings_object('ctrl')
data_settings = s.get_settings_object('data')
#  print(ctrl_settings)
print(data_settings)
