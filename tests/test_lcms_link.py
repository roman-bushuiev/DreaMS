import numpy as np
import pandas as pd

from dreams.utils.lcms import link_ms2_to_features


def _features(rows):
    return pd.DataFrame(rows, columns=["feature_id", "mz_apex", "rt_start_s", "rt_end_s"])


def test_a_feature_containing_the_scan_beats_a_closer_mz_reached_through_the_margin():
    # a small feature ending 5 s before the scan, its m/z 0.2 ppm closer, and the large feature the scan lies in
    feats = _features([(0, 217.086258, 563.5, 569.4), (1, 217.086212, 569.7, 697.5)])
    assert link_ms2_to_features(np.array([217.0863]), np.array([574.4]), feats, tol_mz_ppm=5, tol_rt_s=5).tolist() == [1]


def test_inside_several_features_the_closest_mz_wins():
    feats = _features([(0, 300.0010, 100, 200), (1, 300.0002, 150, 160)])
    assert link_ms2_to_features(np.array([300.0]), np.array([155.0]), feats, tol_mz_ppm=10, tol_rt_s=5).tolist() == [1]


def test_only_through_the_margin_the_nearest_range_wins_and_the_accepted_set_is_unchanged():
    feats = _features([(0, 300.0000, 100, 110), (1, 300.0005, 113, 120), (2, 300.0000, 200, 210)])
    got = link_ms2_to_features(np.array([300.0, 300.0, 300.0, 400.0]), np.array([111.5, 216.0, 150.0, 105.0]), feats,
                               tol_mz_ppm=10, tol_rt_s=5)
    assert got.tolist() == [0, -1, -1, -1]          # 1.5 s past feature 0 beats 1.5 s before feature 1 by m/z;
    got = link_ms2_to_features(np.array([300.0]), np.array([112.4]), feats, tol_mz_ppm=10, tol_rt_s=5)
    assert got.tolist() == [1]                      # 0.6 s from feature 1 is nearer than 2.4 s from feature 0
    got = link_ms2_to_features(np.array([300.0]), np.array([215.0]), feats, tol_mz_ppm=10, tol_rt_s=5)
    assert got.tolist() == [2]                      # exactly at the margin still links, as before


def test_apex_only_features_keep_their_window():
    feats = pd.DataFrame({"feature_id": [0], "mz_apex": [300.0], "rt_apex_s": [100.0]})
    got = link_ms2_to_features(np.array([300.0, 300.0]), np.array([109.0, 111.0]), feats, tol_mz_ppm=10, tol_rt_s=5)
    assert got.tolist() == [0, -1]                  # apex +- tol, widened by tol: 90..110 s
