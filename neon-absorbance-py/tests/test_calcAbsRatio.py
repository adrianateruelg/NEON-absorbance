import pytest
import pandas as pd
from neon_absorbance import calcAbsRatio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_df(wavelengths, absorbances, n_samples=1, sample_id="s1"):
    """Build a minimal absorbance DataFrame for testing."""
    return pd.DataFrame({
        "sampleID": [sample_id] * len(wavelengths),
        "domainID": ["D01"] * len(wavelengths),
        "siteID": ["SITE"] * len(wavelengths),
        "collectDate": ["2024-01-01"] * len(wavelengths),
        "wavelength": wavelengths,
        "decadicAbsorbance": absorbances,
    })


# ---------------------------------------------------------------------------
# Core ratio tests
# ---------------------------------------------------------------------------

def test_calcAbsRatio_even_wavelengths():
    """Even wavelengths are selected directly; ratio = mean(0.5,0.6) / mean(0.2,0.3) = 0.55/0.25 = 2.2."""
    df = _base_df([300, 300, 400, 400], [0.5, 0.6, 0.2, 0.3])

    result = calcAbsRatio(df, 300, 400)

    assert result["absRatio"].iloc[0] == pytest.approx(2.2)


def test_calcAbsRatio_odd_wavelength1():
    """Odd wavelength1: neighbours 300 and 302 are averaged for wavelength 301."""
    df = _base_df([300, 302, 400, 400], [0.5, 0.6, 0.2, 0.3])

    result = calcAbsRatio(df, 301, 400)

    assert result["absRatio"].iloc[0] == pytest.approx(2.2)


def test_calcAbsRatio_odd_wavelength2():
    """Odd wavelength2: neighbours 400 and 402 are averaged for wavelength 401."""
    df = _base_df([300, 300, 400, 402], [0.5, 0.6, 0.2, 0.3])

    result = calcAbsRatio(df, 300, 401)

    assert result["absRatio"].iloc[0] == pytest.approx(2.2)


# ---------------------------------------------------------------------------
# Fe correction tests
# ---------------------------------------------------------------------------

def test_calcAbsRatio_fe_correction_columns():
    """Fe correction path produces an absRatioCorrected column."""
    df = _base_df([300, 400], [0.5, 0.2])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["Fe"],
        "analyteConcentration": [1.0],
    })

    result = calcAbsRatio(df, 300, 400, concentrationData=concentration, correctFe=True)

    assert "absRatioCorrected" in result.columns


def test_calcAbsRatio_fe_correction_value():
    """Fe correction produces the correct numerical ratio.

    At λ=300, Fe=1: correction = (-0.00000044*90000) - (0.00007755*300) + 0.11337671
                                 = -0.0396 - 0.023265 + 0.11337671 = 0.05051071
    corrected_300 = 0.5 - 0.05051071 = 0.44948929

    At λ=400, Fe=1: correction = (-0.00000044*160000) - (0.00007755*400) + 0.11337671
                                 = -0.0704 - 0.03102 + 0.11337671 = 0.01195671
    corrected_400 = 0.2 - 0.01195671 = 0.18804329

    absRatioCorrected = 0.44948929 / 0.18804329 ≈ 2.3904
    """
    df = _base_df([300, 400], [0.5, 0.2])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["Fe"],
        "analyteConcentration": [1.0],
    })

    result = calcAbsRatio(df, 300, 400, concentrationData=concentration, correctFe=True)

    assert result["absRatioCorrected"].iloc[0] == pytest.approx(2.3903447445532353, rel=1e-6)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_calcAbsRatio_output_columns():
    """Output has exactly the expected columns when correctFe=False."""
    df = _base_df([300, 400], [0.5, 0.2])

    result = calcAbsRatio(df, 300, 400)

    assert list(result.columns) == ["domainID", "siteID", "sampleID", "collectDate", "absRatio"]


def test_calcAbsRatio_output_columns_with_fe():
    """Output has exactly the expected columns when correctFe=True."""
    df = _base_df([300, 400], [0.5, 0.2])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["Fe"],
        "analyteConcentration": [1.0],
    })

    result = calcAbsRatio(df, 300, 400, concentrationData=concentration, correctFe=True)

    assert list(result.columns) == [
        "domainID", "siteID", "sampleID", "collectDate", "absRatio", "absRatioCorrected"
    ]


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_calcAbsRatio_empty_data_raises():
    df = pd.DataFrame(columns=["sampleID", "domainID", "siteID", "collectDate",
                                "wavelength", "decadicAbsorbance"])
    with pytest.raises(ValueError, match="No absorbance data"):
        calcAbsRatio(df, 300, 400)


def test_calcAbsRatio_wavelength1_too_low_raises():
    df = _base_df([300, 400], [0.5, 0.2])
    with pytest.raises(ValueError, match="Wavelength 1"):
        calcAbsRatio(df, 100, 400)


def test_calcAbsRatio_wavelength1_too_high_raises():
    df = _base_df([300, 400], [0.5, 0.2])
    with pytest.raises(ValueError, match="Wavelength 1"):
        calcAbsRatio(df, 700, 400)


def test_calcAbsRatio_wavelength2_too_low_raises():
    df = _base_df([300, 400], [0.5, 0.2])
    with pytest.raises(ValueError, match="Wavelength 2"):
        calcAbsRatio(df, 300, 100)


def test_calcAbsRatio_wavelength2_too_high_raises():
    df = _base_df([300, 400], [0.5, 0.2])
    with pytest.raises(ValueError, match="Wavelength 2"):
        calcAbsRatio(df, 300, 700)


def test_calcAbsRatio_correctFe_no_concentration_raises():
    df = _base_df([300, 400], [0.5, 0.2])
    with pytest.raises(ValueError, match="No concentration data"):
        calcAbsRatio(df, 300, 400, correctFe=True)


def test_calcAbsRatio_correctFe_no_fe_analyte_raises():
    """concentrationData present but contains no Fe rows."""
    df = _base_df([300, 400], [0.5, 0.2])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["DOC"],
        "analyteConcentration": [5.0],
    })
    with pytest.raises(ValueError, match="No Fe concentration data"):
        calcAbsRatio(df, 300, 400, concentrationData=concentration, correctFe=True)
