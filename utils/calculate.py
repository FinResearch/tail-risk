# TODO: replace scipy dependency with statistics module from stdlib
#  import scipy.stats as st

from settings import settings as s
# TODO: rid dependency on settings module?

import utils
#  import data_io
import plpva as plpva

#  import plot_funcs.tail_risk_plotter as trp

#  distribution_list = ("truncated_power_law", "exponential", "lognormal")
pl_distro_map = {'tpl': "truncated_power_law",
                 'exp': "exponential",
                 'logn': "lognormal"}


def set_tail_series(tail_dir, series):
    return series if tail_dir == 'right' else -series


def calc_n_store(series, results):

    fit = utils.get_fit(series)

    pl_fit = fit.power_law

    results['α'].append(pl_fit.alpha)

    xmin = pl_fit.xmin
    results['xmin'].append(xmin)

    results['σ'].append(pl_fit.sigma)

    results['abs_len'].append(len(series[series >= xmin]))

    #  bound_delta = s.alpha_quantile * s_err
    #  up_bound = alpha + bound_delta
    #  low_bound = alpha - bound_delta

    #  abs_len = len(tail[tail >= xmin])
    #  # TODO: rel_len is just abs_len / const known in advance
    #  rel_len = len(tail[tail >= xmin]) / len(tail)

    # TODO: think of way to rid dependency on settings module
    ks_pv, _ = plpva.plpva(series, xmin, "reps", s.plpva_iter, "silent")
    results['ks_pv'].append(ks_pv)

    for distro, pdf in pl_distro_map.items():
        R, p = fit.distribution_compare("power_law", pdf,
                                        normalized_ratio=True)
        # TODO/ASK: store as (R, p) for each PDF & display adjacently?
        results[f'loglR_{distro}'].append(R)
        results[f'loglp_{distro}'].append(p)
        #  logl_stats['ratio'][pdf].append(R)
        #  logl_stats['pval'][pdf].append(p)

    #  loglr = daily_ratio
    #  loglpv = daily_p

    #  return alpha, up_bound, low_bound, abs_len, rel_len, ks_pv, loglr, loglpv


def calc_extra_plot_data(results):
    extras = ('up_bound', 'low_bound', 'rel_len')
    pass


def build_csv_row(results):
    csv_row = []

    for lab, val in results.items():
        if s.tail_selected == 'both':
            csv_row.append(val[-2])
        csv_row.append(val[-1])

    return csv_row
