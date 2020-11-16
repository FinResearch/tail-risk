import os
import sys

import cli
from cli.gui import GUI

from utils.analysis import analyze_tail
from utils.settings import Settings as Settings

if __name__ == "__main__":
    input_args = GUI().get_cli_input_args()
    # FIXME: multiple GUI boxes popup when using nproc >1
    user_inputs = cli.get_user_inputs.main(args=input_args,
                                           standalone_mode=False)

    settings = Settings(user_inputs).settings

    f = open(os.devnull, 'w')
    sys.stderr = f

    analyze_tail(settings)
