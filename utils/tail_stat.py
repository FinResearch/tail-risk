


def get_tail_stats(fit_obj, tail_data, ks_pvgof_tup):
    alpha = fit_obj.power_law.alpha
    xmin = fit_obj.power_law.xmin
    s_err = fit_obj.power_law.sigma
    tail_size = len(tail_data[tail_data >= xmin])
    ks_pv = ks_pvgof_tup[0]
    return alpha, xmin, s_err, tail_size, ks_pv


# NOTE: do the right/left values have to alternate in output CSV?
def tail_stat_zipper(tstat1, tstat2):
    return [val for pair in zip(tstat1, tstat2) for val in pair]


# TODO: merge this extraction function into get_tail_stats()
def get_logl_tstats(daily_log_ratio, daily_log_pv):
    r_tpl, r_exp, r_logn = daily_log_ratio
    p_tpl, p_exp, p_logn = daily_log_pv
    return list((r_tpl, r_exp, r_logn, p_tpl, p_exp, p_logn))
