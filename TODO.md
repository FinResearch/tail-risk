# main.py
** main.py|22 col 3| # TODO: apply black styling to all modules (i.e. ' --> ")
main.py|24 col 3| # TODO: annotate/de-annotate NOTE notes
** main.py|25 col 3| # TODO: move imports needed conditionally to within those branches,
main.py|27 col 3| # TODO: remove needless assertion stmt(s) after code is well-tested



# utils/

## _plpva.py
*** |125 col 19| # TODO: look into numpy nansum, nanprod, nan-etc. functions
** |135 col 23| # TODO: the following Python for-loop should be vectorized

## analysis.py
|16 col 15| import sys  # TODO: remove after debugging uses done
|28 col 11| # TODO: factor setting of these boolean flags into own method
|51 col 11| # TODO: factor out repetitive log? (static: date, dynamic: group_label)
*** |77 col 42| "xmin data by file")  # TODO: add file support for -a static?
** |120 col 15| # TODO: try compute ks_pv using MATLAB engine & module, and time
** |161 col 11| # TODO: use np.ndarray instead of pd.Series (wasteful) --> order later
*** |166 col 15| # TODO: consider using pd.DataFrame.replace(, inplace=True) instead
*** |167 col 15| # TODO: can also order stats results first, then assign to DF row
**** |182 col 33| def _analyze_next(self):  # TODO: combine _analyze_next & _analyze_iter??
**** |204 col 11| # TODO: https://stackoverflow.com/a/52596590/5437918 (use shared DBDFs)
**** |207 col 11| # TODO: look into Queue & Pipe for sharing data
**** |209 col 15| # TODO checkout .map alternatives: .imap, .map_async, etc.
**** |210 col 50| restup_ls = [restup for restup in  # TODO: optimize chunksize below
*** |213 col 11| # TODO: update results_df more efficiently, ex. pd.DataFrame.replace(),
** |222 col 11| # TODO: add other conditions for analyze_sequential (ex. -a static)
**** |231 col 11| # TODO: final clean ups of DF for presentation:
** |252 col 41| def _set_curr_input_array(self):  # TODO: pass curr_iter_id as arg???
** |268 col 7| # TODO: consider vectorizing operations on all tickers
** |269 col 41| def _set_curr_input_array(self):  # TODO: pass curr_iter_id as arg???

## results.py
**** |41 col 11| # TODO: add col_idx name 'category' for moments, tails, tstat, logl lvl
*** |49 col 7| # TODO look into pd.concat alternatives
**** |72 col 11| # TODO: merge the 2x4 block containing label 'moments'

## returns.py
**** |13 col 11| # TODO/FIXME: remove redundant/useless assertion statements
|29 col 11| # TODO: move above printing into verbosity logging
|49 col 11| # TODO: can add below info as DEBUG logging
**** |88 col 11| # TODO: implement std/abs for when target is 'tail' in individual mode
** |90 col 11| # TODO optimize below: no need to .flatten() when not analyze_group??
*** |108 col 36| if self.sr.standardize:  # TODO: calc always if getting moments here
**** |125 col 13| # FIXME/TODO: implement std/abs when target is 'tail' in individual mode
*** |141 col 36| if self.sr.standardize:  # TODO: calc always if getting moments here
***** |146 col 15| # TODO/FIXME: add standardization for return window w/ NaN values
*** |181 col 36| if self.sr.standardize:  # TODO: calc always if getting moments here

## settings.py
** |16 col 15| # TODO: coord. w/ attrs.yaml to mark some opts private w/ _-prefix
** |37 col 7| # TODO: test & improve (also maybe add __repr__?)
|38 col 41| def __str__(self, subsett=None):  # TODO: make part of --verbose logging??
|48 col 46| print(val.info())  # TODO: use try-except on df.info?
*** L56: return ''  # TODO/FIXME: instead of just print, should return proper str val
*** |79 col 15| # TODO: consider performing averaging of xmins from passed file
** |176 col 54| #  if self.partition_group_leftovers:  # TODO: opt not yet usable
|184 col 11| # TODO: use slice+loc to org ticks then set toplvl-idx to rm pd depend
*** |188 col 11| # TODO look into pd.concat alternatives
*** |201 col 53| cfg_fpath = 'config/output_columns.yaml'  # TODO: improve pkg/path sys
|323 col 11| # TODO: add below printing to appropriate verbosity logging
**** |346 col 11| # TODO: refactor above and below to more DRY form
*** |362 col 21| pass  # TODO: implement this case, where both types of cols are present
*** |377 col 50| SETTINGS_CFG = 'config/settings.yaml'  # TODO: refactor PATH into root
**** |396 col 29| class GroupingName(str):  # TODO: add allowed values constraint when have time
*** |416 col 3| # TODO: move these Enum types into a constants.py module?
*** |424 col 7| # TODO: just subclass Enum, and return PERIOD.value for self.labelstep??



# cli/

## __init__.py
*** |4 col 3| # TODO: add ROOT_DIR to sys.path in entrypoint/top-level when packaged
|15 col 3| # TODO: customize --help
|21 col 68| default=f'{ROOT_DIR}/db_tests/dbMarkitST.xlsx')  # TODO: remove default after completion
|26 col 3| # TODO: add opts: '--multicore', '--interative', '--gui' (using easygui),
|28 col 33| #                 '--verbose' # TODO: use count opt for -v?
|29 col 52| #  @click.option('-v', '--verbose', count=True)  # TODO: save -v for --verbose?
|36 col 3| # TODO add subcommands: plot & resume_calculation (given full/partial data),
|41 col 3| # TODO:hash opts to uniq-ID usr-input; useful for --partial-saves & --load-opts
**** |51 col 3| # TODO: log to stdin analysis progress (ex. 7/89 dates for ticker XYZ analyzed)

## gui.py
*** |6 col 48| OPT_CFG_DIR = f'{ROOT_DIR}/config/options/'  # TODO: use pathlib.Path ??
**** |74 col 15| # TODO: add try-except for file-opening to handle wrong filetype

## options.py
*** |16 col 48| OPT_CFG_DIR = f'{ROOT_DIR}/config/options/'  # TODO: use pathlib.Path ??
*** related to the above TODO: use importlib.resources to properly package
|40 col 26| else:  # TODO: revise error message
|101 col 7| # TODO: make index_col case-insensitive? i.e. 'Date' or 'date'
*** |107 col 3| # TODO: optimize using list.index(value)?
*** |120 col 53| '[defaults: (66, 0)]\n')  # TODO: don't hardcode dflt
*** |137 col 60| appr_args.help += f'  [defaults: ({lkbk_dflt}, 1)]'  # TODO: don't hrdcode
* |140 col 3| # TODO: checkout default_map & auto_envvar_prefix for click.Context
* |142 col 3| # TODO: subsume func under approach_args CB, since xmin_args no longer uses it?
*** |188 col 11| # TODO: confirm w/ click --> dash for negative num now works?!?
*** |192 col 11| # TODO/FIXME: modify click.Option to accept '-' as -ve args on the CLI
** |218 col 3| # TODO: create & send PR implementing this type of feature below to Click??
** |228 col 3| # TODO: consider shoving _customize_show_default_boolcond into wrapper below??
*** |244 col 7| # TODO: consider move above mapping to own config
|250 col 3| # TODO: make top-level ctx attr for srcs to all options for convenience??
*** |258 col 15| # TODO: use warnings.showwarning() to write to sys.stdout??
* |265 col 3| # TODO: present possible DB_FILE options if not passed & no defaults set
*** |326 col 20| else:  # FIXME/TODO: this branch will never get reached, no? -> remove?
* |366 col 7| # TODO: consider instead of read file & return DF, just return file handle?
*** |400 col 3| # TODO: consider using cb to limit tau to 1 if approach is monthly & warn
** |446 col 11| # TODO: account for when only 1 num arg passed --> make it window-size?
*** |466 col 15| # ASK/TODO: use '<=' OR is '<' is okay?? i.e. open or closed bounds
* |581 col 11| # TODO/FIXME: use the collections.namedtuple construction above?

## _vnargs.py
|22 col 39| self.min_nargs = min_nargs  # TODO: support 0 args? i.e. treat as flag
|41 col 19| # TODO: also break on getting max_nargs? (allows pos_arg after)
|61 col 19| # TODO: use Click's BadOptionUsage(UsageError) as in the above
|65 col 19| # TODO: choose join separator to ensure no conflict w/ self.sep



# config/

## settings.yaml
|68 col 3| # TODO: add sub-setting for output data formatting?
|69 col 3| # TODO: consider adding a sub-setting section for verbosity logging?


## options/

### group_defaults.yaml
|18 col 3| # TODO: --tickers like option for selecting group(s) to analyze (i.e. specify 'DE' when in country)
|38 col 11| # help: # TODO: customize --help for group?

### attributes.yaml
|28 col 3| # TODO: change opt to 'labels', i.e. tickers for -I, groups for -G
|29 col 13| tickers:  # TODO: allow passing in a .txt file of tickers?
|36 col 15| default:  # TODO: remove default after project completion
|46 col 3| # TODO: use specialized Date type (ex. pd.DatetimeIndex) for correct metavar
*** |112 col 17| - 'none'  # TODO: this should just run individual analysis mode
*** |133 col 24| # - 'delta'  # ASK/TODO: isn't 'delta' more fitting varname?
* |144 col 35| - '--standardize/--no-std'  # TODO: shorten to get --help output onto one line
** |150 col 3| # TODO: create alias -N, --norm for std + abs ??
|266 col 3| # TODO: make VnargsOption work with flags, then combine --ks-iter & --ks-run?
** |283 col 3| # TODO: hide control & plotting options from --help unless specified by some subcommand/flag (to reduce clutter)

### easygui.yaml
|5 col 3| # TODO: unify input choices for GUI & CLI systems when appropriate
**** |9 col 32| box_type: 'fileopenbox'  # TODO: add try-except for file-opening to handle wrong filetype
* L198: data_is_continuous:  # TODO: hide this option??
*** add GUI box for getting returns-statistics or not
** when selecting "manual" OR "percent", ask user if they want to use file for variable values OR fixed value





# plot_funcs/
plot_funcs/boxplot.py|30 col 21| #  + lab  # TODO: add this label
plot_funcs/boxplot.py|38 col 11| # TODO: implement plot saving functionality?
plot_funcs/fits_dict.py|67 col 3| # TODO: implement template inheritance for common fields (ax_table pos, etc.)
plot_funcs/fits_dict.py|130 col 3| # TODO: consider just doing all this directly in a subclass?
plot_funcs/fits_dict.py|155 col 6| #  # TODO: consider making a function that checks for this automatically
plot_funcs/fits_dict.py|159 col 3| # TODO: move the below into own file
plot_funcs/tail_risk_plotter.py|1 col 3| # TODO: need to add boxplot, plotter using powerlaw's Fit API, barplot
plot_funcs/tail_risk_plotter.py|3 col 24| from abc import ABC  # TODO: need to label @abstracmethod to work as ABC
plot_funcs/tail_risk_plotter.py|15 col 3| # TODO: consider making class for these, and pass them as dict objects
plot_funcs/tail_risk_plotter.py|18 col 3| # TODO: consider making values returned from this function part
plot_funcs/tail_risk_plotter.py|20 col 3| # TODO: alternatively, make this into @staticmethod of TimeRollingPlotter
plot_funcs/tail_risk_plotter.py|44 col 3| # TODO: consider moving plotter state into own class
plot_funcs/tail_risk_plotter.py|64 col 11| # TODO: consider passing in only the data needed by the given plot_type
plot_funcs/tail_risk_plotter.py|66 col 11| # TODO: make validator function for plot_type?
plot_funcs/tail_risk_plotter.py|80 col 7| # TODO: also add __repr__ method
plot_funcs/tail_risk_plotter.py|83 col 11| # TODO: flesh this method out with better output
plot_funcs/tail_risk_plotter.py|100 col 7| # TODO: consider moving this method into child class
plot_funcs/tail_risk_plotter.py|107 col 15| # TODO: set fig_name to state "both tails" instead of either R/L
plot_funcs/tail_risk_plotter.py|110 col 34| tails = ("left",)  # TODO: this was to test "left" works
plot_funcs/tail_risk_plotter.py|144 col 11| # TODO: simplify naming to use either 'pos/neg' OR 'right/left'???
plot_funcs/tail_risk_plotter.py|148 col 11| # TODO: below will be diff from self.ticker once unnested in tickers
plot_funcs/tail_risk_plotter.py|149 col 43| self.curr_ticker = self.ticker  # TODO: will be diff when plot unnested
plot_funcs/tail_risk_plotter.py|150 col 11| # TODO: consider adding if-check, to only update self.curr_ptinfo
plot_funcs/tail_risk_plotter.py|165 col 11| # TODO: make template_map subclass specific attribute?
plot_funcs/tail_risk_plotter.py|186 col 11| # TODO: refactor below to be more concise and DRY
plot_funcs/tail_risk_plotter.py|203 col 9| TODO: it should not care about the data being plotted nor the opts
plot_funcs/tail_risk_plotter.py|206 col 11| # TODO: use fig, ax = plt.subplots() idiom to Initialize?
plot_funcs/tail_risk_plotter.py|217 col 11| # TODO: factor this into own function to keep DRY for histogram
plot_funcs/tail_risk_plotter.py|223 col 19| # TODO: ensure all vecs are 2-tups for x-y plot??? (ex. hist)
plot_funcs/tail_risk_plotter.py|227 col 15| # TODO: get line_style from self.curr_ptinfo first?
plot_funcs/tail_risk_plotter.py|255 col 11| # TODO: support interative modes
plot_funcs/tail_risk_plotter.py|263 col 15| # TODO: implement plot saving functionality here
plot_funcs/tail_risk_plotter.py|266 col 7| # TODO: add *methods parameters to be optionally called?
plot_funcs/tail_risk_plotter.py|306 col 11| # TODO: is for-loop necessary if each hist only contains a single vec?
plot_funcs/tail_risk_plotter.py|313 col 15| # TODO: ASK why use: h = 2*IQR/cuberoot(n_vec)
plot_funcs/tail_risk_plotter.py|315 col 15| # TODO: xlim also uses max & min --> keep DRY
plot_funcs/tail_risk_plotter.py|323 col 11| # TODO: factor this into own function to keep DRY for histogram
plot_funcs/tail_risk_plotter.py|357 col 11| # TODO: attach Table object to self?
plot_funcs/tail_risk_plotter.py|379 col 33| self.ax.legend()  # TODO: make legend & grid DRY
plot_funcs/tail_risk_plotter.py|396 col 11| # TODO: consider making fits_dict flat in plot_types level
plot_funcs/tail_risk_plotter.py|408 col 3| # TODO: save a set of results data to do quick plot development!!!
