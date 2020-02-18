import numpy as np
import pandas as pd

from settings import settings as s


# TODO: need to allow Template replacement
csv_filenames = {
    "static": "TailStatistics_Overall.csv",
    "rolling_tailanal": "TailStatistics_504_d=1_pn_normalized_{tck}_KS.csv",
    "rolling_group_ta": "TailStatistics_{el}MA_KS_d=66.csv",
                 }


headers_A = [
    "Positive Tail Exponent",
    "Negative Tail Exponent",
    "Positive Tail xmin",
    "Negative Tail xmin",
    "Positive Tail S.Err",
    "Negative Tail S.Err",
    "Positive Tail Size",
    "Negative Tail Size",
    "Positive Tail KS p-value",
    "Negative Tail KS p-value",
]


headers_B = [
    "LL Ratio Right Tail TPL",
    "LL Ratio Right Tail Exp",
    "LL Ratio Right Tail LogN",
    "LL p-value Right Tail TPL",
    "LL p-value Right Tail Exp",
    "LL p-value Right Tail LogN",
    "LL Ratio Left Tail TPL",
    "LL Ratio Left Tail Exp",
    "LL Ratio Left Tail LogN",
    "LL p-value Left Tail TPL",
    "LL p-value Left Tail Exp",
    "LL p-value Left Tail LogN"
]


def read_settings():
    pass


# TODO: account for GroupTailAnalysis too
def _get_csv_metainfo():

    if s.approach == "Static":
        fname = csv_filenames["static"]
        column_headers = ["Input"] + headers_A
    else:  # NOTE: approach is "Rolling" or "Increasing"
        fname = csv_filenames["rolling_tailanal"]
        column_headers = ["Date"] + headers_A + headers_B

    return fname, column_headers


def write_csv_stats(stats_matrix):

    fname, column_headers = _get_csv_metainfo()
    date_colvec = np.array(s.dates).reshape(len(s.dates), 1)
    df_data = np.hstack((date_colvec, stats_matrix))
    df = pd.DataFrame(df_data, columns=column_headers)

    df.to_csv(fname, index=False)
