import yaml
import easygui

from . import ROOT_DIR
OPT_CFG_DIR = f'{ROOT_DIR}/config/options/'  # TODO: use pathlib.Path ??

#  if __name__ == '__main__':
#      specs_fpath = '../config/options/easygui.yaml'


def _load_easygui_specs():
    specs_fpath = OPT_CFG_DIR + 'easygui.yaml'
    with open(specs_fpath, encoding='utf8') as cfg:
        gui_specs = yaml.load(cfg, Loader=yaml.SafeLoader)

    meta_specs_map = {'box_type': None, 'init_cond': 'True', }  # 'callafter'

    msm_for_gui = {opt: {ms: spec.pop(ms, sd) for
                         ms, sd in meta_specs_map.items()}
                   for opt, spec in gui_specs.items()}

    return gui_specs, msm_for_gui


def get_user_inputs():
    user_inputs = {}
    gui, msm = _load_easygui_specs()
    for opt, specs in gui.items():
        box_type = getattr(easygui, msm[opt]['box_type'])
        if eval(msm[opt]['init_cond']):
            user_inputs[opt] = box_type(**specs)
    return user_inputs
