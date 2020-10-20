import sys

import cli
from utils.analysis import analyze_tail
from utils.settings import Settings
from utils.plotting import plot_ensemble

user_inputs = cli.get_user_inputs.main(standalone_mode=False)

if user_inputs == 0:  # for catching the --help option
    sys.exit()

# create option flag to just print out the user_inputs
#  print(user_inputs)

if __name__ == "__main__":
    #  settings = Settings(user_inputs)
    settings = Settings(user_inputs).settings
    #  print(settings)
    #
    results = analyze_tail(settings)
    plot_ensemble(settings, results)


# TODO: apply black styling to all modules (i.e. ' --> ")
# TODO: annotate/de-annotate NOTE notes
# TODO: move imports needed conditionally to within those branches,
#       example: like warnings.warn & itertools.product in settings.py
# TODO: remove needless assertion stmt(s) after code is well-tested
# alternatively: create custom Assertion that's only in effect w/ some --debug
