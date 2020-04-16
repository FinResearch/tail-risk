from cli import gui

from utils.analysis import analyze_tail
from utils.settings import Settings as Settings

user_inputs = gui.get_user_inputs()

settings = Settings(user_inputs).settings
analyze_tail(settings)
