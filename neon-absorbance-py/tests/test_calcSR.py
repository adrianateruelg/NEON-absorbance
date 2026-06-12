import pytest
import numpy as np
import pandas as pd
from neon_absorbance import calcSR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(wavelengths, absorbances, sample_id="s1"):
    """Build a minimal absorbance DataFrame."""
    return pd.DataFrame({
        "sampleID":          [sample_id] * len(wavelengths),
        "domainID":          ["D01"] * len(wavelengths),
        "siteID":            ["SITE"] * len(wavelengths),
        "collectDate":       ["2024-01-01"] * len(wavelengths),
        "wavelength":        wavelengths,
        "decadicAbsorbance": absorbances,
    })


def _expected_slope(wavelengths, absorbances):
    """Compute the negated slope of log(absorbance) ~ wavelength."""
    slope = np.polyfit(wavelengths, np.log(absorbances), 1)[0]
    return -slope


def _band_wavelengths(lo, hi, step=2):
    return list(range(lo, hi + 1, step))


# Build synthetic test data: absorbance = exp(-k * wavelength)
# Use different k values for each band so SR != 1.
K275 = 0.010   # steeper slope in 275-295 band
K350 = 0.005   # shallower slope in 350-400 band

WL275 = _band_wavelengths(276, 294)   # even wavelengths 276–294
WL350 = _band_wavelengths(350, 400)   # even wavelengths 350–400

ABS275 = [np.exp(-K275 * w) for w in WL275]
ABS350 = [np.exp(-K350 * w) for w in WL350]

ALL_WL  = WL275 + WL350
ALL_ABS = ABS275 + ABS350


# ---------------------------------------------------------------------------
# Core SR tests
# ---------------------------------------------------------------------------

def test_calcSR_basic():
    """SR = slope275 / slope350 matches numpy polyfit directly."""
    df = _make_df(ALL_WL, ALL_ABS)

    result = calcSR(df)

    expected_s275 = _expected_slope(WL275, ABS275)
    expected_s350 = _expected_slope(WL350, ABS350)
    expected_SR = expected_s275 / expected_s350

    assert result["SR"].iloc[0] == pytest.approx(expected_SR, rel=1e-6)


def test_calcSR_sr_is_positive():
    """SR should always be positive for typical absorbance data."""
    df = _make_df(ALL_WL, ALL_ABS)
    result = calcSR(df)
    assert result["SR"].iloc[0] > 0


def test_calcSR_replicate_scans_averaged():
    """Duplicate scans at each wavelength are averaged before fitting."""
    wl_doubled  = ALL_WL + ALL_WL
    abs_doubled = ALL_ABS + ALL_ABS   # identical duplicates → same mean
    df = _make_df(wl_doubled, abs_doubled)

    result_single   = calcSR(_make_df(ALL_WL, ALL_ABS))
    result_replicate = calcSR(df)

    assert result_replicate["SR"].iloc[0] == pytest.approx(
        result_single["SR"].iloc[0], rel=1e-6
    )


# ---------------------------------------------------------------------------
# Fe correction tests
# ---------------------------------------------------------------------------

def test_calcSR_fe_correction_columns():
    """Fe correction path produces an SRCorrected column."""
    df = _make_df(ALL_WL, ALL_ABS)
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte":  ["Fe"],
        "analyteConcentration": [0.1],
    })

    result = calcSR(df, concentrationData=concentration, correctFe=True)

    assert "SRCorrected" in result.columns


def test_calcSR_fe_correction_value():
    """SRCorrected matches numpy polyfit on Fe-corrected absorbances."""
    from neon_absorbance._helpers import _fe_absorbance

    df = _make_df(ALL_WL, ALL_ABS)
    fe_conc = 0.1
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte":  ["Fe"],
        "analyteConcentration": [fe_conc],
    })

    result = calcSR(df, concentrationData=concentration, correctFe=True)

    # Compute expected corrected absorbances manually
    wl275 = np.array(WL275, dtype=float)
    wl350 = np.array(WL350, dtype=float)
    abs275_corr = np.array(ABS275) - _fe_absorbance(fe_conc, wl275)
    abs350_corr = np.array(ABS350) - _fe_absorbance(fe_conc, wl350)

    expected_s275 = _expected_slope(wl275, abs275_corr)
    expected_s350 = _expected_slope(wl350, abs350_corr)
    expected_SRc  = expected_s275 / expected_s350

    assert result["SRCorrected"].iloc[0] == pytest.approx(expected_SRc, rel=1e-6)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_calcSR_output_columns():
    df = _make_df(ALL_WL, ALL_ABS)
    result = calcSR(df)
    assert list(result.columns) == ["domainID", "siteID", "sampleID", "collectDate", "SR"]


def test_calcSR_output_columns_with_fe():
    df = _make_df(ALL_WL, ALL_ABS)
    concentration = pd.DataFrame({
        "sampleID": ["s1"], "analyte": ["Fe"], "analyteConcentration": [0.1],
    })
    result = calcSR(df, concentrationData=concentration, correctFe=True)
    assert list(result.columns) == [
        "domainID", "siteID", "sampleID", "collectDate", "SR", "SRCorrected"
    ]


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_calcSR_empty_data_raises():
    df = pd.DataFrame(columns=["sampleID", "domainID", "siteID", "collectDate",
                                "wavelength", "decadicAbsorbance"])
    with pytest.raises(ValueError, match="No absorbance data"):
        calcSR(df)


def test_calcSR_correctFe_no_concentration_raises():
    df = _make_df(ALL_WL, ALL_ABS)
    with pytest.raises(ValueError, match="No concentration data"):
        calcSR(df, correctFe=True)


def test_calcSR_correctFe_no_fe_analyte_raises():
    df = _make_df(ALL_WL, ALL_ABS)
    concentration = pd.DataFrame({
        "sampleID": ["s1"], "analyte": ["DOC"], "analyteConcentration": [5.0],
    })
    with pytest.raises(ValueError, match="No Fe concentration data"):
        calcSR(df, concentrationData=concentration, correctFe=True)
