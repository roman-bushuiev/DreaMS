import pandas as pd
from enum import Enum, auto, unique
import statistics as stats
import numpy as np
import contextlib
import io as std_io
import fcntl
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple
with contextlib.redirect_stderr(std_io.StringIO()):
    import pyopenms as pyms
from collections import Counter
import dreams.utils.spectra as su
import dreams.utils.misc as utils


@unique
class SpecType(Enum):
    """
    Enum representing the type of spectrum (centroid, profile and other corner cases).
    """

    CENTROID = auto()
    PROFILE = auto()
    THRESHOLDED = auto()
    UNKNOWN = auto()
    SIZE_OF_SPECTRUMTYPE = auto()  # from pyopenms (https://github.com/OpenMS/OpenMS/blob/develop/src/pyOpenMS/pxds/SpectrumSettings.pxd#L56)
    INVALID = auto()


@unique
class MSLevelsOrder(Enum):
    """
    Enum representing order of spectra in MS file.

    Further l_i denotes MS level of i-th spectrum.
    """

    # No spectra
    EMPTY = auto()

    # File contains only one spectrum
    # e.g. [1]
    SINGLE_MS1 = auto()
    # e.g. [2]
    SINGLE_MSN = auto()

    # File misses MS1 precursor spectra
    MISSING_MS1 = auto()

    # For all i: l_i = l_(i-1)
    # e.g. [1, 1, 1, 1]
    UNIFORM_MS1 = auto()
    # e.g. [2, 2, 2]
    UNIFORM_MSN = auto()

    # For all i: l_i = l_(i-1) or
    #            l_i > l_(i-1) and l_i - l_(i-1) = 1 or
    #            l_i < l_(i-1) and l_i = 1
    # and not UNIFORM_MS1 and not UNIFORM_MSN
    # e.g. [1, 2, 2, 3, 3, 1, 2]
    CONSEQUENT_MSN = auto()

    # For all i: l_i = l_(i-1) or
    #            l_i > l_(i-1) and l_i - l_(i-1) = 1 or
    #            l_i < l_(i-1) and l_(i-1) - l_1 = 1
    # and not UNIFORM_MS1 and not UNIFORM_MSN
    # e.g. [1, 2, 3, 2, 3, 1, 2]
    MIXED_MSN = auto()

    # Invalid
    # e.g. [-1, 2, 3]
    # e.g. [1, 3]
    INVALID = auto()

    # Other
    #OTHER = auto()


def get_order_of_spectra(msdata) -> MSLevelsOrder:

    # No spectra
    if not msdata.getSpectra():
        return MSLevelsOrder.EMPTY

    ms_levels = [spectrum.getMSLevel() for spectrum in msdata]

    # Check that all MS levels are positive integers
    for ms_level in ms_levels:
        if not isinstance(ms_level, int) or ms_level < 1:
            return MSLevelsOrder.INVALID

    if len(ms_levels) == 1:
        if ms_levels[0] == 1:
            return MSLevelsOrder.SINGLE_MS1
        else:
            return MSLevelsOrder.SINGLE_MSN

    # Check that MS1 is present
    if 1 not in ms_levels:
        return MSLevelsOrder.MISSING_MS1

    # Go over all pairs of subsequent MS levels and classify
    # their difference to MSLevelOrder's
    pairwise_orders = set()
    for level1, level2 in zip(ms_levels[:-1], ms_levels[1:]):

        if level1 == level2:
            if level1 == 1:
                pairwise_orders.add(MSLevelsOrder.UNIFORM_MS1)
            else:
                pairwise_orders.add(MSLevelsOrder.UNIFORM_MSN)
        elif level2 < level1:
            if level2 == 1:
                pairwise_orders.add(MSLevelsOrder.CONSEQUENT_MSN)
            else:
                pairwise_orders.add(MSLevelsOrder.MIXED_MSN)
        else:  # level2 > level1:
            if level2 - level1 == 1:
                pairwise_orders.add(MSLevelsOrder.CONSEQUENT_MSN)
            else:
                return MSLevelsOrder.INVALID

    if MSLevelsOrder.UNIFORM_MS1 in pairwise_orders and len(pairwise_orders) == 1:
        return MSLevelsOrder.UNIFORM_MS1
    elif MSLevelsOrder.UNIFORM_MSN in pairwise_orders and len(pairwise_orders) == 1:
        return MSLevelsOrder.UNIFORM_MSN
    #elif ms_levels[0] != 1:  # NOTE: possible [2, 2, 3, 3] is ok
    #    return MSLevelsOrder.INVALID
    elif MSLevelsOrder.MIXED_MSN in pairwise_orders:
        return MSLevelsOrder.MIXED_MSN
    else:
        return MSLevelsOrder.CONSEQUENT_MSN


def get_tight_xics(msdata, mz_tol_1=0.5, mz_tol_2=0.01, intensity_rel_tol=0.1, xic_len_thld=5, n_highest_peaks=3):
    """
    Tight XIC at given m/z is a cut of ms data accross rt dimension, containing highest peak and all peaks in its
    neighbourhood wrt rt. Length of the neighbourhood is defined independently in each direction by m/z and intensity
    tolerance parameters. When algorigtmm builds XIC it starts from some particular peak (xic_mz, xic_in) and consequently
    examines peaks in its neighbourhood peak by peak. Suppose (prev_mz, prev_in) and (next_mz, next_in) are two peaks compared
    during the run, where (prev_mz, prev_in) is a current border of the neighbourhood, then the neighbourhood will be extended on
    (next_mz, next_in) only if it satisfies two conditions:
        1) abs(next_mz - xic_mz) <= m/z tolerance
        2) next_in <= prev_in * intensity tolerance

    Algorithm performs 2 traversals accross all MS1 spectra:
        I. Builds tight XICs for m/z's of n_highest_peaks highest peaks of each spectrum, where m/z
            tolerance windown is "wide" (mz_tol_1).
        II. Computes medians of m/z's accross XICs obtained in step I., which are used to build new tight
            XICs with "smaller" m/z tolerance window (mz_tol_2).

    :param msdata: ms data to boild XICs from
    :param mz_tol_1: absolute width of m/z tolerance windown for I. traversal
    :param mz_tol_2: absolute width of m/z tolerance windown for II. traversal
    :param intensity_rel_tol: peaks 
    :param xic_len_thld: threshold for the number of peaks in XICs (XICs are filtered both after I. and II.)
    :param n_highest_peaks: number of highest peaks to choose in I.

    NOTE: Since such XICs contain all peaks in the neighbourhood, they are refered to as tight XICs.

    TODO: improve speed, very slow.
    """

    ms1_spectra = [spectrum for spectrum in msdata if spectrum.getMSLevel() == 1]

    # I. First traversal
    # Build XIC for m/z of each base peak

    xics = []
    xics_mzs = []

    for i in range(len(ms1_spectra)):
        spectrum = ms1_spectra[i]

        highest_peaks = su.get_highest_peaks(spectrum.get_peaks(), n_highest_peaks)

        for xic_mz, xic_in in highest_peaks:

            # NOTE: 1.5 m/z: small enough to capture many distinct m/z's
            # and high enough not to capture distinct m/z's together
            if utils.contains_similar(xics_mzs, xic_mz, 1.5):
                continue

            # Add the peak to final XIC
            xic = [(xic_mz, xic_in, spectrum.getRT())]

            last_intensity = xic_in
            # Search "same" (up to mz_tol) m/z values in previous spectra
            for j in reversed(range(i)):

                prev_spectrum = ms1_spectra[j]
                if len(prev_spectrum.get_peaks()[0]) == 0:
                    break

                mz, intensity = su.get_closest_mz_peak(prev_spectrum.get_peaks(), xic_mz)

                if abs(xic_mz - mz) > mz_tol_1 or intensity < last_intensity * intensity_rel_tol:
                    break
                else:
                    xic.insert(0, (mz, intensity, prev_spectrum.getRT()))

            last_intensity = xic_in
            # Search "same" (up to mz_tol) m/z values in next spectra
            for j in range(i + 1, len(ms1_spectra)):

                next_spectrum = ms1_spectra[j]
                if len(next_spectrum.get_peaks()[0]) == 0:
                    break

                mz, intensity = su.get_closest_mz_peak(next_spectrum.get_peaks(), xic_mz)

                if abs(xic_mz - mz) > mz_tol_1 or intensity < last_intensity * intensity_rel_tol:
                    break
                else:
                    xic.append((mz, intensity, next_spectrum.getRT()))

            xics.append(np.array(xic).T)
            xics_mzs.append(xic_mz)

    # Filter out xics having less than xic_len_thld peaks
    xics = [xic for xic in xics if len(xic[0]) >= xic_len_thld]
    xics1 = xics

    # II. Second traversal
    # Build new XICs based on median values of previous XICs

    median_mzs = [stats.median(xic[0]) for xic in xics]

    # 1. Find highest peaks of the new XICs
    # Mzs close to median_mzs but with highest intensities across the whole msdata
    highest_peaks = []
    for median_mz in median_mzs:

        # m/z, intensity, i
        highest_peak = -1, -1, -1
        for i, spectrum in enumerate(ms1_spectra):

            if len(spectrum.get_peaks()[0]) == 0:
                continue

            mz, intensity = su.get_closest_mz_peak(spectrum.get_peaks(), median_mz)
            if intensity > highest_peak[1] and abs(median_mz - mz) < mz_tol_1:
                highest_peak = mz, intensity, i

        if highest_peak[2] != -1:
            highest_peaks.append(highest_peak)

    # 2. Build new (tight) XICs: go left and right from highest peaks
    xics = []
    for highest_peak in highest_peaks:
        xic_mz = highest_peak[0]
        xic_in = highest_peak[1]
        i = highest_peak[2]

        xic = [(highest_peak[0], highest_peak[1], ms1_spectra[i].getRT())]

        last_intensity = xic_in
        # Search "same" (up to mz_tol) m/z values in previous spectra
        for j in reversed(range(i)):

            prev_spectrum = ms1_spectra[j]
            if len(prev_spectrum.get_peaks()[0]) == 0:
                break

            mz, intensity = su.get_closest_mz_peak(prev_spectrum.get_peaks(), xic_mz)

            if abs(xic_mz - mz) > mz_tol_2 or intensity < last_intensity * intensity_rel_tol:
                break
            else:
                xic.insert(0, (mz, intensity, prev_spectrum.getRT()))

        last_intensity = xic_in
        # Search "same" (up to mz_tol) m/z values in next spectra
        for j in range(i + 1, len(ms1_spectra)):

            next_spectrum = ms1_spectra[j]
            if len(next_spectrum.get_peaks()[0]) == 0:
                break

            mz, intensity = su.get_closest_mz_peak(next_spectrum.get_peaks(), xic_mz)

            if abs(xic_mz - mz) > mz_tol_2 or intensity < last_intensity * intensity_rel_tol:
                break
            else:
                xic.append((mz, intensity, next_spectrum.getRT()))

        xics.append(np.array(xic).T)

    # Filter out xics having less than xic_len_thld peaks
    xics = [xic for xic in xics if len(xic[0]) >= xic_len_thld]

    return xics1, xics


def sorted_by_rt(msdata):
    return utils.is_sorted([s.getRT() for s in msdata])


def sort_by_rt(msdata):
    return msdata.setSpectra(sorted(msdata.getSpectra(), key=lambda s: s.getRT(), reverse=True))

def remove_electromagnetic_spectra(msdata):
    filtered_spectra = [spectrum for spectrum in msdata if not spectrum.getMetaValue('lowest observed wavelength')]
    msdata.setSpectra(filtered_spectra)
    return msdata

def get_instrument_props(msdata):
    try:
        xics1, xics = get_tight_xics(msdata)
        xics_stdev = [stats.stdev(xic[0]) for xic in xics]

        quality_props = {
            'instrument name': msdata.getInstrument().getName(),
            '#TBXICs(1)': len(xics1),
            '#TBXICs': len(xics),
            'TBXICs mean stdev': stats.mean(xics_stdev) if xics_stdev else None,
            'TBXICs median stdev': stats.median(xics_stdev) if xics_stdev else None
        }
        return quality_props
    except Exception as e:
        print(f'WARNING: Could not calculate instrument properties: {e}')
        return {
            'instrument name': 'Unknown',
            '#TBXICs(1)': -1,
            '#TBXICs': -1,
            'TBXICs mean stdev': None,
            'TBXICs median stdev': None
        }


def get_pwiz_stats(msdata):
    """
    Checks the presence of spectra centroided by ProteoWizard msconvert yet having zero intensities. Outputs the number
    of such spectra and the histogram of types of spectra converted by msconvert.
    """

    pwiz_stats = Counter()
    for i, spectrum in enumerate(msdata):
        for dp in spectrum.getDataProcessing():
            pwiz = 'proteowizard' in dp.getSoftware().getName().lower()
            conversion_mzml = pyms.ProcessingAction.CONVERSION_MZML in dp.getProcessingActions()
            if pwiz and conversion_mzml:
                spec_type = get_spectrum_type(spectrum)
                pwiz_stats['pwiz_to_mzml_type={}'.format(spec_type.value)] += 1
                peaks = spectrum.get_peaks()
                if spec_type == SpecType.CENTROID and peaks and np.count_nonzero(peaks[1] == 0):
                    pwiz_stats[f'pwiz_zero_mz_centroid'] += 1
    return pwiz_stats


def get_spectrum_type(spec: pyms.MSSpectrum, to_int=False) -> SpecType:

    if spec is None:
        return None
    pyopenms_type = spec.getType()
    if pyopenms_type is None:
        return None

    if pyopenms_type == pyms.SpectrumSettings.SpectrumType.UNKNOWN:  # 0 enum int
        spec_type = SpecType.UNKNOWN
    elif pyopenms_type == pyms.SpectrumSettings.SpectrumType.CENTROID:  # 1 enum int
        spec_type = SpecType.CENTROID
    elif pyopenms_type == pyms.SpectrumSettings.SpectrumType.PROFILE:  # 2 enum int
        spec_type = SpecType.PROFILE
    elif pyopenms_type == pyms.SpectrumSettings.SpectrumType.SIZE_OF_SPECTRUMTYPE:  # 3 enum int
        spec_type = SpecType.SIZE_OF_SPECTRUMTYPE
    else:
        spec_type = SpecType.INVALID

    return spec_type.value if to_int else spec_type


def estimate_peak_list_type(pl: np.array, to_int=True, verbose=False):
    """
    Reproduced from MZmine.
    https://github.com/mzmine/mzmine3/blob/master/src/main/java/io/github/mzmine/util/scans/ScanUtils.java#L609

    ASSUMES PEAK LIST TO BE SORTED BY M/Z (no check in favor of performance).
    """

    peaks_n = su.get_num_peaks(pl)
    if verbose:
        print('Num. peaks:', peaks_n)
    if peaks_n < 5:
        return SpecType.CENTROID.value if to_int else SpecType.CENTROID

    mzs = pl[0]
    intensities = pl[1]

    bp_mz, bp_in, bp_i = su.get_base_peak(pl, return_i=True)
    bp_min_i, bp_max_i = su.get_peak_intens_nbhd(pl, bp_i, bp_in / 2, intens_thld_below=False)

    bp_span = bp_max_i - bp_min_i + 1
    bp_mz_span = mzs[bp_max_i] - mzs[bp_min_i]
    mz_span = mzs[-1] - mzs[0]
    if verbose:
        print('Size of base peak span:', bp_span)
        print(f'Base peak m/z span: {bp_mz_span:.2f}')
        print(f'M/z span: {mz_span:.2f} (0.1% of m/z span: {mz_span / 1000:.2f})')
    if bp_span < 3 or bp_mz_span > mz_span / 1000:
        spec_type = SpecType.CENTROID
    else:
        if (intensities == 0).any():
            spec_type = SpecType.PROFILE
        else:
            spec_type = SpecType.THRESHOLDED

    return spec_type.value if to_int else spec_type


# ---------------------------------------------------------------------------
# SIRIUS 6 lcms-align integration (LC-MS1 feature detection).
#
# Wraps `sirius lcms-align --no-align` per mzML and exposes the resulting
# features as a pandas DataFrame with SIRIUS quality categories
# (PEAK / ISOTOPE / MS2 / ADDUCT) on a 5-state DataQuality scale.
#
# Source of truth for the quality enum mapping:
#   sirius-ms/sirius:chemistry_base/.../utils/DataQuality.java
# ---------------------------------------------------------------------------


@unique
class AcquisitionMode(Enum):
    DDA_CENTROID = "DDA_CENTROID"
    DDA_PROFILE = "DDA_PROFILE"
    DIA = "DIA"
    UNKNOWN = "UNKNOWN"


@unique
class QualityCategory(Enum):
    """SIRIUS DataQuality 5-state grade — integer encoding for HDF5 storage."""

    NOT_APPLICABLE = 0
    LOWEST = 1
    BAD = 2
    DECENT = 3
    GOOD = 4

    @classmethod
    def from_sirius(cls, label) -> "QualityCategory":
        """Map a SIRIUS quality label (str or int) to our enum."""
        if label is None or (isinstance(label, float) and np.isnan(label)):
            return cls.NOT_APPLICABLE
        if isinstance(label, (int, np.integer)):
            return cls(int(label))
        s = str(label).strip().upper()
        mapping = {
            "NOT_APPLICABLE": cls.NOT_APPLICABLE, "NA": cls.NOT_APPLICABLE,
            "": cls.NOT_APPLICABLE,
            "LOWEST": cls.LOWEST, "MINOR": cls.LOWEST,
            "BAD": cls.BAD, "UNUSABLE": cls.BAD,
            "DECENT": cls.DECENT, "REGULAR": cls.DECENT,
            "GOOD": cls.GOOD,
        }
        if s not in mapping:
            raise ValueError(f"Unknown SIRIUS quality label: {label!r}")
        return mapping[s]


# Single source of truth: instrument family → {ppm_default, keywords, skip}.
#   * ppm_default: recommended MS2→feature linkage tolerance (None for families
#     SIRIUS cannot process, e.g. low-resolution QQQ).
#   * keywords: lowercase substrings that map an instrument-name string to this
#     family. First family with a matching keyword wins (FAMILY_MATCH_ORDER).
#   * skip: if set, files of this family are short-circuited before lcms-align
#     with this skip_reason (SIRIUS requires accurate-mass data).
#
# Keyword sources: common Thermo / Waters / Bruker / Sciex / Agilent metadata
# strings, the instrument dictionary from Anal. Chem. 2025 (10.1021/acs.analchem.5c06256),
# and MassBank's AC$INSTRUMENT_TYPE controlled vocabulary
# (<Separation>-<Ionization>-<Analyzer>, e.g. LC-ESI-QTOF / LC-ESI-ITFT / LC-ESI-QQ).
INSTRUMENT_FAMILIES = {
    "orbitrap": {
        "ppm_default": 5.0,
        "skip": None,
        "keywords": [
            "orbitrap", "exactive", "exploris", "fusion", "tribrid",
            "lumos", "elite", "velos pro", "ascend",
            # activation-mode-disambiguated Velos/Lumos (HCD → Orbitrap-grade)
            "lc-esi-hcd", "esi-hcd", "hcd velos", "hcd lumos",
            # Anal. Chem. 2025 + MassBank hybrids: q-FT / IT-FT / HF
            "qft", "itft", "ftms", "ft-ms", " hf", "q exactive", "q-exactive",
            "lc-esi-itft", "lc-esi-qft", "esi-itft", "esi-qft",
        ],
    },
    "fticr": {
        "ppm_default": 5.0,
        "skip": None,
        "keywords": ["fticr", "ft-icr", "ft icr", "solarix", "scimax"],
    },
    "qtof": {
        "ppm_default": 15.0,
        "skip": None,
        "keywords": [
            "qtof", "q-tof", "q tof", "qtfo", "ttof", "tripletof", "triple tof",
            "synapt", "xevo", "x500r", "timstof", "maxis", "compact",
            "impact", "impact hd", "microtof", "lct micromass",
            "agilent 6545", "agilent 6550", "agilent 6560", "waters",
            # ion-trap-TOF hybrids classified as qtof per Anal. Chem. 2025
            "ittof", "it-tof",
            # MassBank AC$INSTRUMENT_TYPE
            "lc-esi-qtof", "esi-qtof", "esi-tof", "lc-esi-tof",
            "maldi-toftof", "jms-s3000", "axima qit", "api qstar",
            "lc-esi-qit;4000q", "fab-ebeb",
        ],
    },
    "qqq": {
        "ppm_default": None,
        "skip": "qqq_low_resolution",
        "keywords": [
            "qqq", "triple quad", "triple-quad", "tsq", "quattro",
            "lc-esi-qq", "esi-qq", "xevo tq", "6470", "6495",
        ],
    },
    "tof": {
        "ppm_default": 25.0,
        "skip": None,
        "keywords": ["tof", "axima", "ultraflex"],
    },
    "iontrap": {
        "ppm_default": 50.0,
        "skip": None,
        "keywords": [
            "ion trap", "ion-trap", "iontrap", "ltq", "amazon", "esquire",
            "hct", "finnigan ltq",
            # CID on Velos/Lumos → ion-trap-grade (check AFTER orbitrap's HCD)
            "lc-esi-cid", "esi-cid", "cid velos", "cid lumos",
            "lc-esi-it", "esi-it",
        ],
    },
    "unknown": {"ppm_default": 20.0, "skip": None, "keywords": []},
}

# Order in which families are tested. Orbitrap before iontrap so HCD-Velos
# wins over the generic "velos"/"ltq" ion-trap match; qqq before qtof so the
# "qq" substring doesn't get shadowed by a "q-tof" partial.
FAMILY_MATCH_ORDER = ["orbitrap", "fticr", "qqq", "qtof", "iontrap", "tof"]


def classify_instrument_family(instrument_name: Optional[str]) -> str:
    """Return the instrument family for an instrument-name string.

    One of ``orbitrap | fticr | qqq | qtof | tof | iontrap | unknown`` based on
    substring keywords in :data:`INSTRUMENT_FAMILIES`, tested in
    :data:`FAMILY_MATCH_ORDER`.
    """
    if not instrument_name:
        return "unknown"
    s = str(instrument_name).lower()
    # Activation-mode disambiguation for trap-Orbitrap hybrids (Velos / Lumos /
    # Fusion): the analyzer that recorded the MS2 depends on the fragmentation
    # mode, not the hardware name. CID → ion-trap-grade; HCD → Orbitrap-grade.
    # This must run before the keyword scan, where "lumos"/"velos" would
    # otherwise match orbitrap.
    is_hybrid = any(k in s for k in ("velos", "lumos", "fusion"))
    if is_hybrid and "cid" in s and "hcd" not in s:
        return "iontrap"
    for family in FAMILY_MATCH_ORDER:
        for kw in INSTRUMENT_FAMILIES[family]["keywords"]:
            if kw in s:
                return family
    return "unknown"


def default_ppm_for_instrument(instrument_name: Optional[str]) -> Optional[float]:
    """Recommended MS2→feature linkage ppm tolerance for an instrument string.

    Empty/unknown → ``INSTRUMENT_FAMILIES['unknown']['ppm_default']``. Returns
    ``None`` for families SIRIUS cannot process (QQQ). To get the matched
    family, use :func:`classify_instrument_family`.
    """
    return INSTRUMENT_FAMILIES[classify_instrument_family(instrument_name)]["ppm_default"]


# pyopenms ActivationMethod enum int → MassBank fragmentation-mode code.
# (MassBank AC$MASS_SPECTROMETRY: FRAGMENTATION_MODE vocabulary.)
#
# IMPORTANT: pyopenms (verified 2026-05-28) maps the PSI-MS CV term
# ``MS:1000422`` ("beam-type collision-induced dissociation" = HCD) to enum
# value 16 internally — even though it *names* enum 16 ``LIFT`` (a MALDI-TOF
# technique). HCD-on-Orbitrap mzMLs come out as 16. We therefore label both
# 14 AND 16 as HCD here; LIFT is unreachable through this code path on
# Orbitrap/QTOF data (which is all we process).
ACTIVATION_METHOD_NAMES = {
    0: "CID", 1: "PSD", 2: "PD", 3: "SID", 4: "BIRD", 5: "ECD",
    6: "IMD", 7: "SORI", 8: "HCD", 9: "LCID", 10: "PHD", 11: "ETD",
    12: "PQD", 13: "TRAP", 14: "HCD", 15: "INSOURCE", 16: "HCD",
}


def activation_modes_from_precursor(prec) -> str:
    """Return a ``;``-joined MassBank fragmentation-mode string for a precursor.

    ``prec`` is a pyopenms ``Precursor``. Empty string if no activation method
    is annotated. Multiple methods (e.g. supplemental activation) are joined.
    """
    try:
        methods = sorted(prec.getActivationMethods())
    except Exception:
        return ""
    codes = []
    for m in methods:
        code = ACTIVATION_METHOD_NAMES.get(int(m))
        if code and code not in codes:
            codes.append(code)
    return ";".join(codes)


_ADDUCT_CHARGE_RE = re.compile(r"\](\d*)([+-])\s*$")


def _parse_adduct_charge(adduct: Optional[str]) -> int:
    """Parse the signed charge from an adduct string.

    ``[M+H]+`` → +1, ``[M+2H]2+`` → +2, ``[M-H]-`` → -1, ``[2M+Na]+`` → +1
    (dimer, still singly charged), ``""`` / unparseable → 0.
    """
    if not adduct:
        return 0
    m = _ADDUCT_CHARGE_RE.search(str(adduct).strip())
    if not m:
        return 0
    n = int(m.group(1)) if m.group(1) else 1
    return n if m.group(2) == "+" else -n


# Monoisotopic atomic masses (Da) for common adduct/loss formulas. The set is
# intentionally small — these are the only elements appearing in SIRIUS's
# canonical adduct strings observed in our corpus (H, C, N, O, Na, K, S, Cl).
_ATOM_MASS = {
    "H": 1.00782503207, "C": 12.0, "N": 14.0030740048, "O": 15.99491461956,
    "S": 31.97207100, "P": 30.97376163, "F": 18.998403163,
    "Cl": 34.96885268, "Na": 22.98976928, "K": 38.96370668,
}
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_ADDUCT_BODY_RE = re.compile(
    r"^\[(\d*)M((?:\s*[+\-]\s*[A-Za-z0-9]+)*)\]\d*[+\-]\s*$"
)


def _formula_mass(formula: str) -> float:
    """Sum of monoisotopic masses for a chemical formula (e.g. 'H3N' -> 17.027)."""
    m = 0.0
    for atom, n in _FORMULA_TOKEN_RE.findall(formula):
        if not atom or atom not in _ATOM_MASS:
            return float("nan")
        m += _ATOM_MASS[atom] * (int(n) if n else 1)
    return m


def parse_adduct_mass_shift(adduct: Optional[str]) -> Tuple[int, float]:
    """Parse a SIRIUS adduct string to ``(multiplicity, mass_shift_Da)``.

    Layout: ``[<n>M (+/- <formula>)*]<z>(+|-)``. ``mass_shift`` is the net atomic
    delta applied to ``nM`` to produce the observed ion (so the neutral mass is
    recovered as ``M = (m/z * |z| - mass_shift) / multiplicity``).

    Returns ``(1, NaN)`` for unparseable input. Uses atom masses for added/lost
    formulas (the ~0.5 mDa proton-vs-H-atom discrepancy is well below any ppm
    grouping tolerance at mass-spec-relevant m/z).
    """
    if not adduct:
        return 1, float("nan")
    m = _ADDUCT_BODY_RE.match(str(adduct).strip().replace(" ", ""))
    if not m:
        return 1, float("nan")
    multiplicity = int(m.group(1)) if m.group(1) else 1
    shift = 0.0
    body = m.group(2)
    # body is like '+H' or '+H3N+H' or '-H4O2+H'. Tokenize signed terms.
    i = 0
    while i < len(body):
        sign = 1 if body[i] == "+" else -1
        i += 1
        j = i
        while j < len(body) and body[j] not in "+-":
            j += 1
        term = body[i:j]
        if term:
            mass = _formula_mass(term)
            if mass != mass:  # NaN
                return 1, float("nan")
            shift += sign * mass
        i = j
    return multiplicity, shift


def assign_compound_ids_by_adduct(
    features_df,
    mz_tol_ppm: float = 10.0,
    rt_tol_s: float = 5.0,
    id_prefix: str = "C",
):
    """Post-hoc compound grouping: cluster features sharing a neutral molecule.

    SIRIUS only populates ``compoundId`` during multi-sample cohort alignment.
    For per-file detection we group post-hoc: features with a confidently
    assigned adduct map to a neutral mass; features whose neutral masses agree
    within ``mz_tol_ppm`` AND whose [rt_start, rt_end] intervals overlap within
    ``rt_tol_s`` are the same compound (e.g. ``[M+H]+`` and ``[M+Na]+`` of one
    molecule eluting together).

    Features WITHOUT a parseable adduct get an empty compound_id (no signal to
    group them — full coverage would need MS2 spectral networking, separate).

    Returns a 1-D ``np.ndarray[object]`` of shape ``(len(features_df),)`` with
    compound_id strings (e.g. ``"C000042"``) or empty strings.
    """
    n = len(features_df)
    out = np.array([""] * n, dtype=object)
    if n == 0:
        return out

    # Compute neutral mass per feature with a parseable adduct.
    adducts = features_df["adduct"].tolist()
    mz = features_df["mz_apex"].to_numpy(dtype=float)
    rt_lo = features_df["rt_start_s"].to_numpy(dtype=float)
    rt_hi = features_df["rt_end_s"].to_numpy(dtype=float)
    z_col = (features_df["adduct_charge"].to_numpy(dtype=int)
             if "adduct_charge" in features_df.columns
             else np.ones(n, dtype=int))
    neutral_mass = np.full(n, np.nan, dtype=float)
    for i, ad in enumerate(adducts):
        if isinstance(ad, (bytes, bytearray)):
            ad = ad.decode("utf-8", "ignore")
        if not ad:
            continue
        mult, shift = parse_adduct_mass_shift(ad)
        if shift != shift:  # NaN
            continue
        z = abs(int(z_col[i])) or 1
        neutral_mass[i] = (mz[i] * z - shift) / mult

    # Sort indices of features with a neutral mass and sweep to form clusters
    # (neutral mass within ppm AND RT interval overlap with ANY current member).
    idx = np.argsort(neutral_mass, kind="stable")
    valid = ~np.isnan(neutral_mass)
    idx = idx[valid[idx]]
    cluster_id_per_feat = np.full(n, -1, dtype=np.int32)
    clusters = []  # list of dicts: {"mass": float, "rt_lo": float, "rt_hi": float}
    for fi in idx:
        m = neutral_mass[fi]
        lo, hi = rt_lo[fi], rt_hi[fi]
        # NaN RT bounds -> skip RT check (rare); fall back to apex window.
        if not (lo == lo and hi == hi):
            lo, hi = -1e18, 1e18
        # Find an existing cluster within mz tolerance whose RT overlaps.
        matched = -1
        for ci, c in enumerate(clusters):
            if abs(c["mass"] - m) / max(m, 1e-9) > mz_tol_ppm * 1e-6:
                continue
            if c["rt_hi"] + rt_tol_s < lo or hi + rt_tol_s < c["rt_lo"]:
                continue
            matched = ci
            # Tighten cluster's mass to the mean; widen RT envelope.
            c["mass"] = (c["mass"] * c["count"] + m) / (c["count"] + 1)
            c["count"] += 1
            c["rt_lo"] = min(c["rt_lo"], lo)
            c["rt_hi"] = max(c["rt_hi"], hi)
            break
        if matched < 0:
            clusters.append({"mass": m, "rt_lo": lo, "rt_hi": hi, "count": 1})
            matched = len(clusters) - 1
        cluster_id_per_feat[fi] = matched

    # Emit only clusters with >= 2 members (singletons aren't a "grouping").
    counts = np.bincount(cluster_id_per_feat[cluster_id_per_feat >= 0],
                         minlength=len(clusters))
    width = max(6, len(str(max(1, len(clusters)))))
    for fi in range(n):
        ci = cluster_id_per_feat[fi]
        if ci >= 0 and counts[ci] >= 2:
            out[fi] = f"{id_prefix}{ci:0{width}d}"
    return out


# Thermo's NCE reference: NCE = eV × 500 / (precursor_mz × |charge|).
_NCE_REF_MASS = 500.0


def normalize_collision_energy(ce_raw, precursor_mz, charge,
                               instrument_family: str) -> Tuple[float, float, str]:
    """Best-effort collision-energy normalization → ``(ev_est, nce_est, unit)``.

    Different vendors report CE in different units:
      * Orbitrap (Thermo) report **NCE** (normalized collision energy, a
        precursor-mass-independent percentage). We convert toward absolute eV.
      * QTOF / TOF (Waters, Bruker, Sciex, Agilent) report **absolute eV**.
        We provide a reverse NCE estimate.
      * Ion-trap CID is low-res; we pass eV through, NCE estimate is NaN.
      * QQQ / unknown family: unit ``'unknown'``, both estimates NaN.

    Conversion (Thermo standard, charge-aware):
        eV  = NCE × precursor_mz × |charge| / 500
        NCE = eV  × 500 / (precursor_mz × |charge|)

    ``ce_raw`` may be a string (e.g. ``"30 eV"``, ``"20;30;40"``) or float.
    Stepped CE (multiple ``;``-separated values) → unit ``'stepped'``, both NaN.
    """
    nan = float("nan")
    # Stepped CE in the raw string?
    if isinstance(ce_raw, str) and ";" in ce_raw:
        return nan, nan, "stepped"
    try:
        ce = float(str(ce_raw).lower().replace("ev", "").replace("nce", "").strip())
    except (TypeError, ValueError):
        return nan, nan, "unknown"
    if ce <= 0:
        return nan, nan, "unknown"

    try:
        pmz = float(precursor_mz)
        z = abs(int(charge)) or 1
    except (TypeError, ValueError):
        pmz, z = nan, 1
    scale = (pmz * z) if (pmz and np.isfinite(pmz)) else nan

    if instrument_family in ("orbitrap", "fticr"):
        # Reported value is NCE.
        ev_est = ce * scale / _NCE_REF_MASS if np.isfinite(scale) else nan
        return float(ev_est), float(ce), "NCE"
    if instrument_family in ("qtof", "tof"):
        # Reported value is absolute eV.
        nce_est = ce * _NCE_REF_MASS / scale if np.isfinite(scale) and scale else nan
        return float(ce), float(nce_est), "eV"
    if instrument_family == "iontrap":
        # CID eV; NCE not meaningful.
        return float(ce), nan, "eV"
    # qqq / unknown
    return nan, nan, "unknown"


def get_instrument_name(mzml_pth) -> str:
    """Lazily read the instrument-name CV term from an mzML.

    Returns the empty string on failure. Uses ``OnDiscMSExperiment`` so only
    the header is parsed (~ms, regardless of file size).
    """
    try:
        exp = pyms.OnDiscMSExperiment()
        if not exp.openFile(str(mzml_pth)):
            return ""
        return str(exp.getExperimentalSettings().getInstrument().getName() or "")
    except Exception:
        return ""


def detect_acquisition_mode(mzml_pth, dia_window_width_threshold_da: float = 8.0,
                            max_spectra_to_probe: int = 200,
                            return_diagnostics: bool = False):
    """Inspect an mzML and classify it as DDA centroid / DDA profile / DIA / unknown.

    SIRIUS lcms-align expects centroided DDA input. This function lets the
    pipeline skip files that would otherwise crash or produce garbage features.

    Lazy: uses ``OnDiscMSExperiment`` so only scan headers (not peak arrays) are
    loaded for the first ``max_spectra_to_probe`` MSn spectra.

    Heuristics:
      * centroid vs profile: pyopenms ``spec.getType()`` over MSn spectra, with
        :func:`estimate_peak_list_type` as a fallback when CV terms are missing.
      * DDA vs DIA: median isolation window width across MSn spectra.
        DIA windows are typically ≥ ``dia_window_width_threshold_da`` Da.

    When ``return_diagnostics`` is True, returns ``(mode, dia_window_width_da)``
    where the width is the median isolation-window width (NaN if unmeasurable).
    """
    def _ret(mode, width=float("nan")):
        return (mode, width) if return_diagnostics else mode

    exp = pyms.OnDiscMSExperiment()
    if not exp.openFile(str(mzml_pth)):
        return _ret(AcquisitionMode.UNKNOWN)

    n_spec = exp.getNrSpectra()
    msn_indices = []
    for i in range(n_spec):
        if exp.getSpectrum(i).getMSLevel() >= 2:
            msn_indices.append(i)
            if len(msn_indices) >= max_spectra_to_probe:
                break
    if not msn_indices:
        return _ret(AcquisitionMode.UNKNOWN)

    # centroid vs profile from CV terms first
    types_cv = Counter()
    for i in msn_indices[:50]:
        s = exp.getSpectrum(i)
        types_cv[get_spectrum_type(s)] += 1
    centroid_cv = types_cv.get(SpecType.CENTROID, 0)
    profile_cv = types_cv.get(SpecType.PROFILE, 0)
    unknown_cv = (types_cv.get(SpecType.UNKNOWN, 0)
                  + types_cv.get(None, 0))

    if unknown_cv > centroid_cv + profile_cv:
        # CV terms unreliable → empirical fallback on first MSn spectrum
        first = exp.getSpectrum(msn_indices[0])
        mzs, intens = first.get_peaks()
        mzs = np.asarray(mzs, dtype=float)
        intens = np.asarray(intens, dtype=float)
        if len(mzs) == 0:
            return _ret(AcquisitionMode.UNKNOWN)
        order = np.argsort(mzs)
        pl = np.vstack([mzs[order], intens[order]])
        est = estimate_peak_list_type(pl, to_int=False)
        is_centroid = est == SpecType.CENTROID
        cv_says_centroid = None  # CV terms unreliable for this file
    else:
        is_centroid = centroid_cv >= profile_cv
        cv_says_centroid = centroid_cv >= profile_cv

    widths = []
    for i in msn_indices:
        s = exp.getSpectrum(i)
        for prec in s.getPrecursors():
            lo = prec.getIsolationWindowLowerOffset()
            up = prec.getIsolationWindowUpperOffset()
            if lo is not None and up is not None:
                widths.append(float(lo) + float(up))
    med_width = float(np.median(widths)) if widths else float("nan")

    if not np.isnan(med_width) and med_width >= dia_window_width_threshold_da:
        return _ret(AcquisitionMode.DIA, med_width)
    mode = (AcquisitionMode.DDA_CENTROID if is_centroid
            else AcquisitionMode.DDA_PROFILE)
    return _ret(mode, med_width)


def normalize_mzml_for_sirius(mzml_pth, out_pth) -> Path:
    """Round-trip an mzML through pyopenms so SIRIUS can parse it.

    Some mzMLs (e.g. certain msConvert / vendor exports — observed on the
    piper & tropicana studies) omit the compression CV parameter in
    ``<binaryDataArray>``. pyopenms tolerates this, but SIRIUS rejects the file
    with ``IllegalStateException: Required compression CV parameter not found in
    BinaryDataArray`` — which then surfaces downstream as an empty project and
    an IndexOutOfBounds in the alignment backbone. Re-storing via pyopenms
    writes a compliant mzML (compression CV terms present), which SIRIUS reads.

    Costs ~3 s + a temp copy; lossless for the spectra/precursor/RT info SIRIUS
    uses. Returns ``out_pth``.
    """
    out_pth = Path(out_pth)
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    exp = pyms.MSExperiment()
    pyms.MzMLFile().load(str(mzml_pth), exp)
    pyms.MzMLFile().store(str(out_pth), exp)
    # Force the bytes to disk before a SIRIUS subprocess opens the file. On
    # Lustre, close-to-open consistency isn't guaranteed unless the writer
    # syncs — without this, lcms-align can open a not-yet-flushed file and
    # silently detect zero features.
    try:
        fd = os.open(str(out_pth), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass
    return out_pth


def run_sirius_lcms_align(mzml_pth, work_dir,
                          log_pth: Optional[Path] = None,
                          sirius_bin: Optional[str] = None,
                          extra_args: Optional[list] = None,
                          timeout_s: Optional[int] = None,
                          workspace=None) -> Path:
    """Run ``sirius lcms-align --no-align`` on a single mzML.

    Returns the path to the resulting ``.sirius`` project file. SIRIUS 6
    writes the project as a single opaque binary file (NitriteDB), not a
    directory.

    ``workspace`` (optional) relocates the SIRIUS workspace (``.rtoken`` +
    caches) so concurrent workers don't race the shared login file.
    """
    mzml_pth = Path(mzml_pth)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # SIRIUS requires the project name (sans .sirius) to match [a-zA-Z0-9_-]+ —
    # no dots or other punctuation. Sanitize so inputs like "x.94936.norm.mzML"
    # don't crash lcms-align with a picocli ParameterException.
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", mzml_pth.stem)
    project_pth = work_dir / f"{safe_stem}.sirius"
    if project_pth.exists():
        project_pth.unlink()

    extra = list(extra_args) if extra_args else []
    args = _sirius_cmd(sirius_bin, workspace,
                       "-i", str(mzml_pth), "-o", str(project_pth),
                       "lcms-align", "--no-align", *extra)

    log_pth = Path(log_pth) if log_pth is not None else work_dir / f"{mzml_pth.stem}.sirius.log"
    with open(log_pth, "w") as logf:
        proc = subprocess.run(args, stdout=logf, stderr=subprocess.STDOUT,
                              timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sirius lcms-align failed for {mzml_pth} "
            f"(returncode={proc.returncode}); see {log_pth}"
        )
    return project_pth


# Columns we read from SIRIUS' `write-summaries --feature-quality-summary` TSV
# (`feature_quality.tsv` in the output directory). Verified empirically against
# SIRIUS 6.3.5 — these are the only LC-MS feature fields exposed at the summary
# layer.
#
# Fields NOT available from this TSV (and therefore stored as NaN/empty in the
# HDF5 features group until we add a REST or project-space parsing path):
#     rt_start, rt_end, area, intensity_apex, adduct identity, isotope pattern
SIRIUS_TSV_COLUMN_MAP = {
    "sirius_feature_id": "alignedFeatureId",
    "mapping_feature_id": "mappingFeatureId",
    "mz_apex": "ionMass",
    "rt_apex_s": "retentionTimeInSeconds",
    "peak_quality": "Peak Quality",
    "alignment_quality": "Alignment Quality",
    "isotope_quality": "Isotope Pattern Quality",
    "ms2_quality": "Fragmentation Pattern Quality",
    "adduct_quality": "Adduct Assignment Quality",
    "overall_quality": "overallFeatureQuality",
}


def _write_summaries(project_pth: Path,
                     out_dir: Path,
                     sirius_bin: Optional[str] = None,
                     log_pth: Optional[Path] = None,
                     workspace=None) -> Path:
    """Materialize per-feature summary TSVs from a SIRIUS .sirius project.

    ``project_pth`` is the single ``.sirius`` file (not a directory) produced by
    ``lcms-align``. ``out_dir`` is the directory where TSVs are written.
    Returns ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = _sirius_cmd(sirius_bin, workspace,
                       "-p", str(project_pth),
                       "write-summaries",
                       "--feature-quality-summary",
                       "--output", str(out_dir))
    log_pth = Path(log_pth) if log_pth is not None else out_dir / "write_summaries.log"
    with open(log_pth, "w") as logf:
        proc = subprocess.run(args, stdout=logf, stderr=subprocess.STDOUT,
                              check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sirius write-summaries failed for {project_pth} "
            f"(returncode={proc.returncode}); see {log_pth}"
        )
    return out_dir


def extract_features(project_pth: Path,
                     sirius_bin: Optional[str] = None,
                     summaries_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read per-feature output of a SIRIUS lcms-align project as a DataFrame.

    ``project_pth`` is the ``.sirius`` file produced by
    :func:`run_sirius_lcms_align`. Columns returned:

      feature_id (0..F-1), sirius_feature_id, external_feature_id, mz_apex,
      rt_apex_s, rt_start_s, rt_end_s, charge, has_ms1, has_ms2, overall_quality,
      adduct (semicolon-joined list, empty if SIRIUS detected none).

    Quality is mapped to integer :class:`QualityCategory` codes (0..4).

    This is the lean, scalar-only DataFrame. Per-feature isotope pattern,
    intensity_apex and area need follow-on calls — see
    :func:`extract_features_with_traces` (TODO).

    Uses the SIRIUS REST API headlessly (started + torn down per call). The TSV
    summary fallback path lives in :func:`extract_features_via_tsv` for
    environments where launching the REST server is undesirable.
    """
    project_pth = Path(project_pth)
    with sirius_rest_server() as base_url:
        project_id = _open_project(base_url, project_pth)
        try:
            feats = _list_aligned_features(base_url, project_id)
        finally:
            _close_project(base_url, project_id)
    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame({
        "sirius_feature_id":   [f.get("alignedFeatureId") for f in feats],
        "external_feature_id": [int(f.get("externalFeatureId", -1)) for f in feats],
        "mz_apex":             [float(f.get("ionMass", float("nan"))) for f in feats],
        "rt_apex_s":           [float(f.get("rtApexSeconds", float("nan"))) for f in feats],
        "rt_start_s":          [float(f.get("rtStartSeconds", float("nan"))) for f in feats],
        "rt_end_s":            [float(f.get("rtEndSeconds", float("nan"))) for f in feats],
        "charge":              [int(f.get("charge", 0)) for f in feats],
        "has_ms1":             [bool(f.get("hasMs1", False)) for f in feats],
        "has_ms2":             [bool(f.get("hasMsMs", False)) for f in feats],
        "overall_quality":     [QualityCategory.from_sirius(f.get("quality")).value
                                for f in feats],
        "adduct":              [";".join(f.get("detectedAdducts") or [])
                                for f in feats],
    })
    df["overall_quality"] = df["overall_quality"].astype(np.int8)
    df["feature_id"] = np.arange(len(df), dtype=np.int32)
    return df


def extract_features_via_tsv(project_pth: Path,
                             sirius_bin: Optional[str] = None,
                             summaries_dir: Optional[Path] = None) -> pd.DataFrame:
    """TSV-only fallback: cheaper than REST but missing rt_start/rt_end/adduct.

    Use this when the REST server cannot be launched (e.g. a worker node
    without SIRIUS-login access). The output uses the column set from
    :data:`SIRIUS_TSV_COLUMN_MAP` (no rt_start/rt_end; quality fields for
    *all five* SIRIUS categories).
    """
    project_pth = Path(project_pth)
    if summaries_dir is None:
        summaries_dir = project_pth.with_suffix(".summaries")
    summaries_dir = Path(summaries_dir)
    tsv = summaries_dir / "feature_quality.tsv"
    if not tsv.exists():
        _write_summaries(project_pth, summaries_dir, sirius_bin=sirius_bin)
    if not tsv.exists():
        return pd.DataFrame(columns=list(SIRIUS_TSV_COLUMN_MAP.keys()) + ["feature_id"])

    raw = pd.read_csv(tsv, sep="\t")
    cols_out = {}
    for our_col, sirius_col in SIRIUS_TSV_COLUMN_MAP.items():
        if sirius_col in raw.columns:
            cols_out[our_col] = raw[sirius_col]
    df = pd.DataFrame(cols_out)
    for q in ("peak_quality", "alignment_quality", "isotope_quality",
              "ms2_quality", "adduct_quality", "overall_quality"):
        if q in df.columns:
            df[q] = df[q].map(QualityCategory.from_sirius).map(lambda x: x.value)
            df[q] = df[q].astype(np.int8)
    df["feature_id"] = np.arange(len(df), dtype=np.int32)
    return df


# ---------------------------------------------------------------------------
# SIRIUS REST helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _sirius_cmd(sirius_bin: Optional[str], workspace, *args) -> list:
    """Build a sirius CLI argv with an optional per-process ``--workspace``.

    ``--workspace`` is a SIRIUS *global* option (before the subcommand) that
    relocates the workspace dir holding ``.rtoken`` + caches. Giving each
    concurrent worker its own workspace avoids the shared-``.rtoken``
    file-write race that invalidates the login under parallelism.
    """
    bin_ = sirius_bin or os.environ.get("SIRIUS_BIN") or "sirius"
    cmd = [bin_]
    if workspace:
        cmd += ["--workspace", str(workspace)]
    cmd += list(args)
    return cmd


def prewarm_sirius_login(sirius_bin: Optional[str] = None,
                         timeout_s: float = 60.0,
                         workspace=None) -> bool:
    """Re-mint the access token from the workspace's ``.rtoken``.

    Runs ``sirius login --show`` in a subprocess. Each SIRIUS CLI invocation
    starts a new JVM that re-authenticates from the persistent refresh token,
    so this is the cheapest way to validate that Bright Giant's auth endpoint
    is reachable AND the refresh token hasn't been revoked.

    Returns True on success, False on any subprocess/timeout error.
    """
    try:
        subprocess.run(
            _sirius_cmd(sirius_bin, workspace, "login", "--show"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout_s, check=True,
        )
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError):
        return False


@contextlib.contextmanager
def sirius_auth_lock(timeout_s: float = 120.0,
                     lock_path: Optional[Path] = None):
    """Serialize SIRIUS auth handshakes across processes via an fcntl lock.

    The lock file lives at ``~/.sirius-6.3/auth.lock`` by default — i.e. on the
    same filesystem as the refresh token. Workers that share ``$HOME`` will
    serialize cleanly; workers on different hosts won't (which is fine — only
    one host's worker fleet needs serialization at a time).

    On timeout, falls through *without* raising — the caller proceeds without
    serialization rather than fail outright.
    """
    if lock_path is None:
        lock_path = Path.home() / ".sirius-6.3" / "auth.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    acquired = False
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(0.25)
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        f.close()


@contextlib.contextmanager
def sirius_rest_server(sirius_bin: Optional[str] = None,
                       port: Optional[int] = None,
                       startup_timeout_s: float = 180.0,
                       shutdown_timeout_s: float = 10.0,
                       serialize_startup: bool = True,
                       workspace=None):
    """Context manager that runs ``sirius rest --headless`` in the background.

    Yields the REST base URL (e.g. ``http://127.0.0.1:9234``). On exit it
    triggers ``/actuator/shutdown`` and kills the subprocess if shutdown
    times out.

    When ``serialize_startup`` is True (default), the JVM boot phase (Popen →
    first ``/api/info`` response) is wrapped in :func:`sirius_auth_lock`, so
    concurrent workers don't boot SIRIUS JVMs simultaneously. The lock is
    released as soon as the server is up — long-running REST queries run
    *outside* the lock, so concurrent extraction stays parallel. This fixes
    the 2-of-4 REST-startup timeouts observed at 4-way concurrency.
    """
    port = port or _free_port()
    base = f"http://127.0.0.1:{port}"

    def _boot_and_wait():
        proc = subprocess.Popen(
            _sirius_cmd(sirius_bin, workspace, "rest", "--headless",
                        "-p", str(port), "--enable-rest-shutdown"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        t0 = time.time()
        while time.time() - t0 < startup_timeout_s:
            try:
                with urllib.request.urlopen(f"{base}/api/info", timeout=2) as r:
                    r.read(1)
                return proc
            except Exception:
                time.sleep(0.5)
        # Timed out — kill the half-booted JVM before raising.
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(
            f"sirius rest service did not come up on {base} in "
            f"{startup_timeout_s}s"
        )

    if serialize_startup:
        with sirius_auth_lock(timeout_s=startup_timeout_s + 60.0):
            proc = _boot_and_wait()
    else:
        proc = _boot_and_wait()

    try:
        yield base
    finally:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/actuator/shutdown", method="POST"),
                timeout=shutdown_timeout_s,
            ).read(1)
        except Exception:
            pass
        try:
            proc.wait(timeout=shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _http_json(method: str, url: str, body: Optional[dict] = None,
               timeout_s: float = 60.0):
    req = urllib.request.Request(url, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=data, timeout=timeout_s) as r:
        raw = r.read()
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))


def _open_project(base_url: str, project_pth: Path,
                  project_id: Optional[str] = None) -> str:
    """Open an existing .sirius project via REST. Returns the projectId used."""
    project_id = project_id or f"p{int(time.time() * 1000)}"
    q = urllib.parse.urlencode({"pathToProject": str(project_pth)})
    _http_json("PUT", f"{base_url}/api/projects/{project_id}?{q}")
    return project_id


def _close_project(base_url: str, project_id: str) -> None:
    try:
        _http_json("DELETE", f"{base_url}/api/projects/{project_id}")
    except Exception:
        pass


def _list_aligned_features(base_url: str, project_id: str,
                           page_size: int = 1000,
                           opt_fields: Optional[list] = None) -> list:
    """Paginate /aligned-features/page and return the concatenated content list."""
    out = []
    page = 0
    while True:
        q = urllib.parse.urlencode({"page": page, "size": page_size})
        if opt_fields:
            q += "&" + urllib.parse.urlencode(
                [("optFields", f) for f in opt_fields])
        d = _http_json(
            "GET",
            f"{base_url}/api/projects/{project_id}/aligned-features/page?{q}",
        )
        out.extend(d.get("content", []))
        if page + 1 >= d.get("page", {}).get("totalPages", 0):
            break
        page += 1
    return out


def get_feature_detail(base_url: str, project_id: str,
                       aligned_feature_id: str,
                       opt_fields: Optional[list] = None) -> dict:
    """Single-feature GET with optFields (e.g. ``['msData']`` for isotope pattern)."""
    q = ""
    if opt_fields:
        q = "?" + urllib.parse.urlencode(
            [("optFields", f) for f in opt_fields])
    return _http_json(
        "GET",
        f"{base_url}/api/projects/{project_id}/aligned-features/{aligned_feature_id}{q}",
    )


def get_feature_traces(base_url: str, project_id: str,
                       aligned_feature_id: str) -> dict:
    """Chromatographic traces (RT axis + intensity arrays per trace) for a feature."""
    return _http_json(
        "GET",
        f"{base_url}/api/projects/{project_id}/aligned-features/{aligned_feature_id}/traces",
    )


def link_ms2_to_features(precursor_mz: np.ndarray,
                         rt_seconds: np.ndarray,
                         features_df: pd.DataFrame,
                         tol_mz_ppm: float = 10.0,
                         tol_rt_s: float = 5.0) -> np.ndarray:
    """For each MS2 spectrum row, return the parent feature_id (or -1).

    Linkage rule:
      precursor m/z within ``tol_mz_ppm`` of feature ``mz_apex`` AND
      MS2 RT within ``[rt_apex - tol_rt_s, rt_apex + tol_rt_s]`` (if
      ``rt_start``/``rt_end`` columns exist, those are used instead).

    If multiple features match, the one with closest m/z (in ppm) wins.
    """
    precursor_mz = np.asarray(precursor_mz, dtype=float)
    rt_seconds = np.asarray(rt_seconds, dtype=float)
    n = precursor_mz.shape[0]
    out = np.full(n, -1, dtype=np.int32)
    if features_df is None or len(features_df) == 0:
        return out

    mz = features_df["mz_apex"].to_numpy(dtype=float)
    # Prefer rt_start/rt_end (suffix or no-suffix); fall back to apex ± tol_rt_s.
    cols = features_df.columns
    if "rt_start_s" in cols and "rt_end_s" in cols:
        rt_lo = features_df["rt_start_s"].to_numpy(dtype=float)
        rt_hi = features_df["rt_end_s"].to_numpy(dtype=float)
    elif "rt_start" in cols and "rt_end" in cols:
        rt_lo = features_df["rt_start"].to_numpy(dtype=float)
        rt_hi = features_df["rt_end"].to_numpy(dtype=float)
    else:
        if "rt_apex_s" in cols:
            apex = features_df["rt_apex_s"].to_numpy(dtype=float)
        elif "rt_apex" in cols:
            apex = features_df["rt_apex"].to_numpy(dtype=float)
        else:
            return out
        rt_lo = apex - tol_rt_s
        rt_hi = apex + tol_rt_s

    fid = features_df["feature_id"].to_numpy(dtype=np.int32)

    for i in range(n):
        pmz, prt = precursor_mz[i], rt_seconds[i]
        if not np.isfinite(pmz) or not np.isfinite(prt):
            continue
        ppm = 1e6 * np.abs(mz - pmz) / np.maximum(pmz, 1e-9)
        in_rt = (prt >= rt_lo - tol_rt_s) & (prt <= rt_hi + tol_rt_s)
        ok = (ppm <= tol_mz_ppm) & in_rt
        if not ok.any():
            continue
        best = np.where(ok, ppm, np.inf).argmin()
        out[i] = int(fid[best])
    return out


def _get_quant_table(base_url: str, project_id: str,
                     quant_type: str = "AREA_UNDER_CURVE") -> dict:
    """Bulk per-feature quantification (area or apex height) for one run.

    SIRIUS exposes per-run quantification as a row-id-keyed table; the values
    array shape is ``(n_features, n_runs)``. For our single-mzML lcms-align
    project ``n_runs == 1``.
    """
    q = urllib.parse.urlencode({"type": quant_type})
    return _http_json(
        "GET",
        f"{base_url}/api/projects/{project_id}/aligned-features/quant-table?{q}",
    )


def compute_features_for_mzml(
    mzml_pth,
    work_dir,
    sirius_bin: Optional[str] = None,
    sirius_port: Optional[int] = None,
    keep_project: bool = False,
    trace_ppm_max: Optional[float] = None,
    min_snr: Optional[float] = None,
    noise_intensity: Optional[float] = None,
    sensitive_mode: bool = False,
    normalize_mzml: bool = True,
    sirius_workspace=None,
):
    """Run SIRIUS lcms-align on an mzML and return a feature DataFrame.

    Pipeline (one mzML):
      1. ``detect_acquisition_mode`` — short-circuit if non-DDA-centroid (we
         still return a structured ``(empty_df, [-1]*n_ms2, skip_reason)``
         result so the caller can write an empty features group).
      2. ``run_sirius_lcms_align`` — produces the ``.sirius`` project file.
      3. ``_write_summaries`` — materializes the per-category quality TSV.
      4. Start ``sirius rest --headless`` (context-managed), open project.
      5. Bulk fetch features via ``/aligned-features/page?optFields=msData``.
      6. Bulk fetch areas via ``/aligned-features/quant-table?type=AREA_UNDER_CURVE``.
      7. Merge REST scalars + REST isotope patterns + TSV per-category quality
         + REST areas into one feature DataFrame.

    Returns a 4-tuple ``(features_df, isotope_patterns, attrs, status)``:

      * ``features_df`` — one row per feature with columns
        ``feature_id (0..F-1), sirius_feature_id, external_feature_id,
        mz_apex, rt_apex_s, rt_start_s, rt_end_s, charge, area,
        has_ms1, has_ms2, peak_quality, alignment_quality, isotope_quality,
        ms2_quality, adduct_quality, overall_quality, adduct``.
      * ``isotope_patterns`` — list[list[dict(mz, intensity)]]; aligned to
        ``features_df["feature_id"]`` rows.
      * ``attrs`` — dict of HDF5 group attributes
        (``sirius_version, lcms_align_args, acquisition_mode``).
      * ``status`` — dict ``{ok: bool, skip_reason: str}``.

    The caller (typically :func:`attach_features_group`) is responsible for
    serializing into HDF5.
    """
    work_dir = Path(work_dir)
    # Assemble optional SIRIUS knobs. Defaults (None / False) → SIRIUS uses its
    # data-driven estimates; only trace_ppm_max is a real user knob.
    extra_args = []
    if trace_ppm_max is not None:
        extra_args += ["--trace-ppm-max", str(trace_ppm_max)]
    if min_snr is not None:
        extra_args += ["--min-snr", str(min_snr)]
    if noise_intensity is not None:
        extra_args += ["--noise-intensity", str(noise_intensity)]
    if sensitive_mode:
        extra_args += ["--sensitive-mode"]
    lcms_args = "--no-align" + ((" " + " ".join(extra_args)) if extra_args else "")

    sirius_input, norm_mzml, attrs, skip = _detection_prelude(
        mzml_pth, work_dir, normalize_mzml, sirius_bin, lcms_align_args=lcms_args)
    if skip is not None:
        return skip

    project_pth = run_sirius_lcms_align(sirius_input, work_dir,
                                        sirius_bin=sirius_bin,
                                        extra_args=extra_args or None,
                                        workspace=sirius_workspace)

    # Per-category quality from TSV (REST only exposes the overall string).
    summaries_dir = project_pth.with_suffix(".summaries")
    if not (summaries_dir / "feature_quality.tsv").exists():
        _write_summaries(project_pth, summaries_dir, sirius_bin=sirius_bin,
                         workspace=sirius_workspace)
    tsv_df = extract_features_via_tsv(project_pth,
                                      sirius_bin=sirius_bin,
                                      summaries_dir=summaries_dir)
    # Build a SIRIUS-id → per-category-quality dict (O(1) lookup; DataFrame.loc
    # is too slow at 22K iterations).
    if "sirius_feature_id" in tsv_df.columns:
        tsv_quality_by_id = {
            str(r.sirius_feature_id): (
                int(getattr(r, "peak_quality", 0)),
                int(getattr(r, "alignment_quality", 0)),
                int(getattr(r, "isotope_quality", 0)),
                int(getattr(r, "ms2_quality", 0)),
                int(getattr(r, "adduct_quality", 0)),
            )
            for r in tsv_df.itertuples(index=False)
        }
    else:
        tsv_quality_by_id = {}

    feats, quant = None, None
    last_err: Optional[Exception] = None
    # With a per-worker ``sirius_workspace`` each process has its own .rtoken,
    # so concurrent auth no longer races a shared file — run fully parallel.
    # Without one, fall back to serializing the whole REST session under the
    # cross-process lock (one live server at a time) to avoid 401s.
    isolated = sirius_workspace is not None
    for attempt in range(2):
        try:
            if isolated:
                prewarm_sirius_login(sirius_bin, workspace=sirius_workspace)
                with sirius_rest_server(sirius_bin=sirius_bin, port=sirius_port,
                                        serialize_startup=False,
                                        workspace=sirius_workspace) as base:
                    project_id = _open_project(base, project_pth)
                    try:
                        feats = _list_aligned_features(base, project_id,
                                                       page_size=500,
                                                       opt_fields=["msData"])
                        quant = _get_quant_table(base, project_id,
                                                 "AREA_UNDER_CURVE")
                    finally:
                        _close_project(base, project_id)
            else:
                with sirius_auth_lock(timeout_s=900.0):
                    prewarm_sirius_login(sirius_bin)
                    with sirius_rest_server(sirius_bin=sirius_bin, port=sirius_port,
                                            serialize_startup=False) as base:
                        project_id = _open_project(base, project_pth)
                        try:
                            feats = _list_aligned_features(base, project_id,
                                                           page_size=500,
                                                           opt_fields=["msData"])
                            quant = _get_quant_table(base, project_id,
                                                     "AREA_UNDER_CURVE")
                        finally:
                            _close_project(base, project_id)
            break
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 401 and attempt == 0:
                # Re-prewarm + retry once with a short backoff. The previous
                # REST server got into a poisoned auth state; the next iteration
                # starts a fresh JVM that will re-auth from .rtoken.
                time.sleep(2.0 + 3.0 * attempt)
                continue
            break
        except Exception as e:
            last_err = e
            break
    if feats is None:
        if not keep_project:
            project_pth.unlink(missing_ok=True)
            if norm_mzml:
                norm_mzml.unlink(missing_ok=True)
            if summaries_dir.exists():
                shutil.rmtree(summaries_dir, ignore_errors=True)
        reason = (f"http_{last_err.code}" if isinstance(last_err, urllib.error.HTTPError)
                  else type(last_err).__name__ if last_err is not None else "unknown")
        return (
            pd.DataFrame(),
            [],
            attrs,
            {"ok": False, "skip_reason": f"rest_error:{reason}"},
        )

    df, isotope_patterns, status = _build_features_df(feats, quant,
                                                       tsv_quality_by_id)

    if not keep_project:
        project_pth.unlink(missing_ok=True)
        if norm_mzml:
            norm_mzml.unlink(missing_ok=True)
        if summaries_dir.exists():
            shutil.rmtree(summaries_dir, ignore_errors=True)

    return df, isotope_patterns, attrs, status


# Maps SIRIUS quality-report category names → the 5-slot per-category quality
# tuple (peak, alignment, isotope, ms2, adduct). Alignment has no REST category
# (it is NOT_APPLICABLE under --no-align), so it stays 0.
_QR_CATEGORY_TO_SLOT = {
    "Peak Quality": 0,
    "Isotope Pattern Quality": 2,
    "Fragmentation Pattern Quality": 3,
    "Adduct Assignment Quality": 4,
}


def _parse_quality_report(qr: Optional[dict]) -> tuple:
    """REST ``/quality-report`` → (peak, alignment, isotope, ms2, adduct) ints.

    Each ``categories[name]`` is ``{categoryName, items, overallQuality}``; we
    take the per-category ``overallQuality`` enum and map it through
    :class:`QualityCategory`.
    """
    vals = [0, 0, 0, 0, 0]
    for name, cat in ((qr or {}).get("categories") or {}).items():
        slot = _QR_CATEGORY_TO_SLOT.get(name)
        if slot is not None and isinstance(cat, dict):
            vals[slot] = QualityCategory.from_sirius(cat.get("overallQuality")).value
    return tuple(vals)


def _build_features_df(feats: list, quant: Optional[dict],
                       quality_by_id: dict):
    """Merge REST feature scalars + isotope patterns + per-category quality +
    areas into the canonical features DataFrame.

    Shared by the CLI (``compute_features_for_mzml``) and REST
    (``compute_features_via_rest``) paths. ``quality_by_id`` maps
    ``sirius_feature_id`` → ``(peak, alignment, isotope, ms2, adduct)`` ints
    (from the TSV in the CLI path, from ``/quality-report`` in the REST path).

    Returns ``(df, isotope_patterns, status)``.
    """
    # SIRIUS-id → area map. quant.values is (F, n_runs); n_runs == 1 here.
    sid_to_area = {}
    if quant and quant.get("values"):
        row_ids = quant.get("rowIds") or []
        for i, sid in enumerate(row_ids):
            row = quant["values"][i] if i < len(quant["values"]) else []
            if row and row[0] not in (None, "NaN"):
                try:
                    sid_to_area[str(sid)] = float(row[0])
                except (TypeError, ValueError):
                    pass

    rows = []
    isotope_patterns = []
    for f in feats:
        sid = str(f.get("alignedFeatureId"))
        msd = f.get("msData") or {}
        ip = (msd.get("isotopePattern") or {}).get("peaks") or []
        pk, al, iso, ms2, ad = quality_by_id.get(sid, (0, 0, 0, 0, 0))
        # detectedAdducts: SIRIUS's ion-identity-network assignment(s). The top
        # entry is the primary adduct; the rest are alternative hypotheses.
        det = list(f.get("detectedAdducts") or [])
        primary_adduct = det[0] if det else ""
        alt_adducts = ";".join(det[1:]) if len(det) > 1 else ""
        rows.append({
            "sirius_feature_id":   sid,
            "external_feature_id": int(f.get("externalFeatureId", -1)),
            "compound_id":         str(f.get("compoundId") or ""),
            "mz_apex":             float(f.get("ionMass", float("nan"))),
            "rt_apex_s":           float(f.get("rtApexSeconds", float("nan"))),
            "rt_start_s":          float(f.get("rtStartSeconds", float("nan"))),
            "rt_end_s":            float(f.get("rtEndSeconds", float("nan"))),
            "charge":              int(f.get("charge", 0)),
            "adduct_charge":       _parse_adduct_charge(primary_adduct),
            "area":                sid_to_area.get(sid, float("nan")),
            "has_ms1":             bool(f.get("hasMs1", False)),
            "has_ms2":             bool(f.get("hasMsMs", False)),
            "peak_quality":        pk,
            "alignment_quality":   al,
            "isotope_quality":     iso,
            "ms2_quality":         ms2,
            "adduct_quality":      ad,
            "overall_quality":     QualityCategory.from_sirius(f.get("quality")).value,
            "adduct":              primary_adduct,
            "alternative_adducts": alt_adducts,
        })
        isotope_patterns.append(ip)

    if not rows:
        # Detection ran but produced no features (rare; e.g. very sparse MS1).
        return (pd.DataFrame(), [],
                {"ok": True, "skip_reason": "no_features_detected"})

    df = pd.DataFrame(rows)
    df["feature_id"] = np.arange(len(df), dtype=np.int32)
    for q in ("peak_quality", "alignment_quality", "isotope_quality",
              "ms2_quality", "adduct_quality", "overall_quality"):
        df[q] = df[q].astype(np.int8)

    # SIRIUS leaves compoundId empty in single-file lcms-align (it only groups
    # features during multi-sample cohort alignment). Run post-hoc grouping by
    # neutral mass + RT so the column is meaningful — covers features with a
    # parseable adduct (~10%); full coverage needs MS2 spectral networking.
    cids = assign_compound_ids_by_adduct(df)
    df["compound_id"] = cids

    return df, isotope_patterns, {"ok": True, "skip_reason": ""}


def _detection_prelude(mzml_pth, work_dir, normalize_mzml, sirius_bin,
                       lcms_align_args="--no-align"):
    """Instrument classification + skip checks + mzML normalization.

    Shared prelude for both feature-detection paths. Returns
    ``(sirius_input, norm_mzml, attrs, skip)`` where ``skip`` is ``None`` to
    proceed, or a ready ``(empty_df, [], attrs, status)`` tuple to return.
    """
    mzml_pth = Path(mzml_pth)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    instrument_name = get_instrument_name(mzml_pth)
    instrument_family = classify_instrument_family(instrument_name)
    attrs = {
        "sirius_version": _get_sirius_version(sirius_bin),
        "lcms_align_args": lcms_align_args,
        "acquisition_mode": "UNKNOWN",
        "instrument_name": instrument_name,
        "instrument_family": instrument_family,
        "auto_tol_mz_ppm": INSTRUMENT_FAMILIES[instrument_family]["ppm_default"],
    }

    skip_reason = INSTRUMENT_FAMILIES[instrument_family].get("skip")
    if skip_reason:
        return None, None, attrs, (pd.DataFrame(), [], attrs,
                                   {"ok": False, "skip_reason": skip_reason})

    mode = detect_acquisition_mode(mzml_pth)
    attrs["acquisition_mode"] = mode.value
    if mode not in (AcquisitionMode.DDA_CENTROID,):
        return None, None, attrs, (pd.DataFrame(), [], attrs,
                                   {"ok": False,
                                    "skip_reason": f"acquisition_mode:{mode.value}"})

    sirius_input = mzml_pth
    norm_mzml = None
    if normalize_mzml:
        try:
            _tmp = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
            norm_mzml = Path(_tmp) / f"{mzml_pth.stem}.{os.getpid()}.norm.mzML"
            normalize_mzml_for_sirius(mzml_pth, norm_mzml)
            sirius_input = norm_mzml
            attrs["mzml_normalized"] = "True"
        except Exception:
            sirius_input = mzml_pth
            attrs["mzml_normalized"] = "False"
    return sirius_input, norm_mzml, attrs, None


def _import_local_files_job(base_url: str, project_id: str,
                            file_paths: list, align: bool = False) -> dict:
    """POST a local-file import (runs lcms-align inside the server). Returns Job.

    ``align=False`` mirrors the CLI ``--no-align`` (per-file detection, no
    cross-sample alignment). The nested Deviation params are left at SIRIUS's
    data-driven defaults — the working CLI path never set them either.
    """
    q = urllib.parse.urlencode({"alignLCMSRuns": str(bool(align)).lower()})
    url = (f"{base_url}/api/projects/{project_id}/import/"
           f"ms-data-local-files-job?{q}&optFields=progress")
    return _http_json("POST", url, body=[str(p) for p in file_paths])


def _poll_job(base_url: str, project_id: str, job_id: str,
              timeout_s: float = 1200.0, poll_s: float = 2.0) -> str:
    """Poll a project job until terminal. Returns the final state string."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        j = _http_json(
            "GET",
            f"{base_url}/api/projects/{project_id}/jobs/{job_id}?optFields=progress",
        )
        st = (j.get("progress") or {}).get("state")
        if st in ("DONE", "FAILED", "CANCELED"):
            return st
        time.sleep(poll_s)
    return "TIMEOUT"


def compute_features_via_rest(
    mzml_pth,
    base_url: str,
    work_dir,
    sirius_bin: Optional[str] = None,
    normalize_mzml: bool = True,
    fetch_quality: bool = True,
    job_timeout_s: float = 1200.0,
):
    """Feature detection through a SHARED persistent ``sirius rest`` server.

    Unlike :func:`compute_features_for_mzml` (which spawns a fresh ``lcms-align``
    CLI + REST server per file, re-authenticating each time), this runs ALL
    SIRIUS work inside one already-authenticated server over HTTP — zero
    per-file auth. That is the only model that survives a long / concurrent run
    (the per-file-auth model invalidates the OAuth token after ~149 files).

    Pipeline (one mzML, against ``base_url``):
      1. ``_detection_prelude`` — instrument skip + mzML normalization.
      2. PUT an empty project; POST ``import/ms-data-local-files-job`` (runs
         lcms-align in the server); poll the job.
      3. ``/aligned-features/page?optFields=msData`` + ``/quant-table``.
      4. ``/quality-report`` per feature → per-category quality.
      5. ``_build_features_df`` (shared merge); DELETE the project.

    Returns the same 4-tuple as :func:`compute_features_for_mzml`.
    """
    sirius_input, norm_mzml, attrs, skip = _detection_prelude(
        mzml_pth, work_dir, normalize_mzml, sirius_bin,
        lcms_align_args="--no-align (rest import)")
    if skip is not None:
        return skip

    project_id = f"f{os.getpid()}_{int(time.time() * 1000) % 10_000_000}"
    proj_tmp = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    project_pth = Path(proj_tmp) / f"{project_id}.sirius"
    feats = quant = None
    last_err = None
    try:
        q = urllib.parse.urlencode({"pathToProject": str(project_pth)})
        _http_json("PUT", f"{base_url}/api/projects/{project_id}?{q}")
        job = _import_local_files_job(base_url, project_id, [sirius_input])
        state = _poll_job(base_url, project_id, job.get("id"),
                          timeout_s=job_timeout_s)
        if state != "DONE":
            return (pd.DataFrame(), [], attrs,
                    {"ok": False, "skip_reason": f"import_{state.lower()}"})
        feats = _list_aligned_features(base_url, project_id, page_size=500,
                                       opt_fields=["msData"])
        quant = _get_quant_table(base_url, project_id, "AREA_UNDER_CURVE")
        quality_by_id = {}
        if fetch_quality:
            for f in feats:
                fid = f.get("alignedFeatureId")
                try:
                    qr = _http_json(
                        "GET",
                        f"{base_url}/api/projects/{project_id}/"
                        f"aligned-features/{fid}/quality-report")
                    quality_by_id[str(fid)] = _parse_quality_report(qr)
                except Exception:
                    pass
    except urllib.error.HTTPError as e:
        last_err = e
    except Exception as e:
        last_err = e
    finally:
        try:
            _http_json("DELETE", f"{base_url}/api/projects/{project_id}")
        except Exception:
            pass
        if norm_mzml:
            Path(norm_mzml).unlink(missing_ok=True)

    if feats is None:
        reason = (f"http_{last_err.code}" if isinstance(last_err, urllib.error.HTTPError)
                  else type(last_err).__name__ if last_err is not None else "unknown")
        return (pd.DataFrame(), [], attrs,
                {"ok": False, "skip_reason": f"rest_error:{reason}"})

    df, isotope_patterns, status = _build_features_df(feats, quant, quality_by_id)
    return df, isotope_patterns, attrs, status


def _get_sirius_version(sirius_bin: Optional[str] = None) -> str:
    """Best-effort `sirius --version`; returns 'unknown' on failure."""
    sirius_bin = sirius_bin or os.environ.get("SIRIUS_BIN") or "sirius"
    try:
        out = subprocess.run([sirius_bin, "--version"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=60)
        for line in out.stdout.decode("utf-8", errors="ignore").splitlines():
            if line.startswith("SIRIUS"):
                return line.strip()
    except Exception:
        pass
    return "unknown"


def attach_features_group(hdf5_pth,
                          features_df: pd.DataFrame,
                          isotope_patterns: list,
                          feature_id_per_ms2: np.ndarray,
                          attrs: Optional[dict] = None,
                          group_name: str = "features",
                          fk_dataset_name: str = "feature_id") -> None:
    """Append a SIRIUS-features group + foreign-key dataset to an existing HDF5.

    Variable-length isotope patterns are stored with the flat+offsets pattern:
    a single concatenated ``isotope_mz_flat`` array plus a cumulative
    ``isotope_offsets`` index of length ``F+1`` such that
    ``isotope_mz_flat[offsets[i]:offsets[i+1]]`` is feature *i*'s pattern.

    Idempotent on a fresh HDF5: re-running deletes and recreates the group.
    """
    import h5py
    hdf5_pth = Path(hdf5_pth)
    attrs = dict(attrs or {})

    n_features = len(features_df)
    n_ms2 = len(feature_id_per_ms2)

    # Build flat+offsets isotope arrays
    if n_features > 0 and isotope_patterns:
        iso_offsets = np.zeros(n_features + 1, dtype=np.int32)
        for i, peaks in enumerate(isotope_patterns):
            iso_offsets[i + 1] = iso_offsets[i] + len(peaks)
        sum_n_iso = int(iso_offsets[-1])
        iso_mz_flat = np.zeros(sum_n_iso, dtype=np.float32)
        iso_in_flat = np.zeros(sum_n_iso, dtype=np.float32)
        pos = 0
        for peaks in isotope_patterns:
            for p in peaks:
                iso_mz_flat[pos] = float(p.get("mz", float("nan")))
                iso_in_flat[pos] = float(p.get("intensity", float("nan")))
                pos += 1
    else:
        iso_offsets = np.zeros(max(n_features + 1, 1), dtype=np.int32)
        iso_mz_flat = np.zeros(0, dtype=np.float32)
        iso_in_flat = np.zeros(0, dtype=np.float32)

    with h5py.File(hdf5_pth, "a") as f:
        # FK column on the MS2 rows (root level).
        if fk_dataset_name in f:
            del f[fk_dataset_name]
        f.create_dataset(fk_dataset_name,
                         data=np.asarray(feature_id_per_ms2, dtype=np.int32),
                         shape=(n_ms2,),
                         compression="gzip", compression_opts=4)

        # Features group.
        if group_name in f:
            del f[group_name]
        g = f.create_group(group_name)

        if n_features > 0:
            g.create_dataset("feature_id",
                             data=features_df["feature_id"].to_numpy(np.int32),
                             compression="gzip", compression_opts=4)
            g.create_dataset(
                "sirius_feature_id",
                data=np.asarray(features_df["sirius_feature_id"].tolist(),
                                dtype=h5py.string_dtype(encoding="utf-8")),
                compression="gzip", compression_opts=4,
            )
            g.create_dataset("external_feature_id",
                             data=features_df["external_feature_id"].to_numpy(np.int32),
                             compression="gzip", compression_opts=4)
            # SIRIUS compound id — groups features that are different ion forms
            # / fragments of the same neutral molecule (ion identity network).
            g.create_dataset(
                "compound_id",
                data=np.asarray(features_df["compound_id"].tolist(),
                                dtype=h5py.string_dtype(encoding="utf-8")),
                compression="gzip", compression_opts=4,
            )
            for col, dtype in [
                ("mz_apex", np.float32), ("rt_apex_s", np.float32),
                ("rt_start_s", np.float32), ("rt_end_s", np.float32),
                ("area", np.float32),
            ]:
                g.create_dataset(col, data=features_df[col].to_numpy(dtype),
                                 compression="gzip", compression_opts=4)
            for col in ("charge", "adduct_charge", "has_ms1", "has_ms2"):
                g.create_dataset(col, data=features_df[col].to_numpy(np.int8),
                                 compression="gzip", compression_opts=4)
            for q in ("peak_quality", "alignment_quality", "isotope_quality",
                      "ms2_quality", "adduct_quality", "overall_quality"):
                g.create_dataset(q, data=features_df[q].to_numpy(np.int8),
                                 compression="gzip", compression_opts=4)
            for col, slen in (("adduct", 32), ("alternative_adducts", 64)):
                g.create_dataset(
                    col,
                    data=np.asarray(features_df[col].tolist(),
                                    dtype=h5py.string_dtype(encoding="utf-8",
                                                           length=slen)),
                    compression="gzip", compression_opts=4,
                )

        # Ragged isotope patterns (flat + offsets).
        g.create_dataset("isotope_mz_flat", data=iso_mz_flat,
                         compression="gzip", compression_opts=4)
        g.create_dataset("isotope_intens_flat", data=iso_in_flat,
                         compression="gzip", compression_opts=4)
        g.create_dataset("isotope_offsets", data=iso_offsets,
                         compression="gzip", compression_opts=4)

        # Group attrs.
        for k, v in attrs.items():
            g.attrs[k] = str(v) if v is not None else ""


def standartize_gnps_species(species: pd.Series):

    # Lowercase
    species = species.str.lower()

    # Add NCBITaxon suffix if known from other entries
    ncbi_suffix = ' (NCBITaxon:'.lower()
    species_to_ncbi = {s.split(ncbi_suffix)[0]: s.split(ncbi_suffix)[1] for s in species.unique().tolist() if isinstance(s, str) and ncbi_suffix in s}
    species = species.apply(lambda s: s if s not in species_to_ncbi else s + ncbi_suffix + species_to_ncbi[s])

    # Manually merge similar species
    species_merged = [
        (['Homo sapiens (NCBITaxon:9606)', 'homo sapiens', 'human', 'Human'], 'Human'),
        (['Mus musculus domesticus', 'Mus musculus (NCBITaxon:10090)', 'Rattus norvegicus (NCBITaxon:10116)', 'Rattus (NCBITaxon:10114)', 'C57BL/6N', 'Mus sp. (NCBITaxon:10095)', 'mice', 'Mice'], 'Mice'),
        (['Ocean Environmental Samples', 'environmental samples <Bacillariophyta> (NCBITaxon:33858)', 'environmental samples <Verrucomicrobiales> (NCBITaxon:48468)', 'environmental samples <delta subdivision> (NCBITaxon:34033)'], 'Environmental')
    ]
    species_merge_map = {}
    for k, v in species_merged:
        for s in k:
            species_merge_map[s.lower()] = v.lower()
    species = species.apply(lambda s: species_merge_map[s] if s in species_merge_map else s)

    # Other
    species = species.rename({'': 'other'})

    return species