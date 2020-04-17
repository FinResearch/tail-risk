import yaml
import easygui

from . import ROOT_DIR
OPT_CFG_DIR = f'{ROOT_DIR}/config/options/'  # TODO: use pathlib.Path ??

#  if __name__ == '__main__':
#      specs_fpath = '../config/options/easygui.yaml'


class GUI:

    def __init__(self):
        self._set_meta_opt_maps()

    def __load_easygui_specs(self):
        specs_fpath = OPT_CFG_DIR + 'easygui.yaml'
        with open(specs_fpath, encoding='utf8') as cfg:
            gui_specs = yaml.load(cfg, Loader=yaml.SafeLoader)
        return ({opt: specs[sptyp] for opt, specs in gui_specs.items()}
                for sptyp in ('metas', 'attrs'))

    def _set_meta_opt_maps(self):
        gui_metas, self.gui_attrs = self.__load_easygui_specs()
        metas = set(mt for mtup in (mt.keys() for opt, mt in gui_metas.items())
                    for mt in mtup)
        for mt in metas:
            mom = {}  # meta option map
            for opt, msp in gui_metas.items():
                if mt in msp:
                    mom[opt] = msp[mt]
            setattr(self, mt, mom)

    def __proc_creation_flags(self, opt, attrs):
        create_gui = True
        flags = self.creation_flags.get(opt)
        if bool(flags):
            for fstr in flags:
                action, cond = fstr.split(': ')
                #  evald_cond = eval(f"getattr(self, '{cond}', None)")

                if action == 'init_on':
                    create_gui = eval(f"self.{cond}")
                elif action.endswith('set_by'):
                    val_attr = action.split('-')[0]
                    assert val_attr in attrs
                    if cond == 'analyze_group':
                        idx = bool(eval(f"self.{cond}"))
                        attrs[val_attr] = attrs[val_attr][idx]
                    elif cond == 'evaluation':
                        import os  # used by eval() to get # processors
                        attrs[val_attr] = eval(attrs[val_attr])
                    else:
                        raise ValueError('this should not be reached!')
        return create_gui

    def _set_bool_flags(self, opt, opt_val):
        bool_flag = self.set_bool_flag.get(opt)
        if bool(bool_flag):
            fname, value = bool_flag
            setattr(self, fname, eval(value))

    def _create_and_run_guis(self):
        uis = {}
        for opt, attrs in self.gui_attrs.items():
            box = getattr(easygui, self.box_type[opt])
            if self.__proc_creation_flags(opt, attrs):
                opt_val = box(**attrs)
                self._set_bool_flags(opt, opt_val)
                uis[opt] = opt_val
        self.user_inputs = uis

    def get_user_inputs(self):
        self._create_and_run_guis()
        return self.user_inputs
