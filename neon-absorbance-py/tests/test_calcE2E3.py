import pytest
import pandas as pd
from neon_absorbance import calcE2E3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_df(wavelengths, absorbances):
    return pd.DataFrame({
        "sampleID":         ["s1"] * len(wavelengths),
        "domainID":         ["D01"] * len(wavelengths),
        "siteID":           ["SITE"] * len(wavelengths),
        "collectDate":      ["2024-01-01"] * len(wavelengths),
        "wavelength":       wavelengths,
        "decadicAbsorbance": absorbances,
    })


# ---------------------------------------------------------------------------
# Core ratio tests
# ---------------------------------------------------------------------------

def test_calcE2E3_basic():
    """E2E3 = abs250 / mean(abs364, abs366) = 0.5 / mean(0.2, 0.3) = 0.5 / 0.25 = 2.0"""
    df = _base_df([250, 364, 366], [0.5, 0.2, 0.3])

    result = calcE2E3(df)

    assert result["E2E3"].iloc[0] == pytest.approx(2.0)


def test_calcE2E3_replicate_scans_averaged():
    """Duplicate scans at each wavelength are averaged before the ratio."""
    df = _base_df(
        [250, 250, 364, 364, 366, 366],
        [0.4, 0.6, 0.1, 0.3, 0.2, 0.4],
    )
    # avg abs250 = 0.5, avg abs364 = 0.2, avg abs366 = 0.3 → avg365 = 0.25
    result = calcE2E3(df)

    assert result["E2E3"].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Fe correction tests
# ---------------------------------------------------------------------------

def test_calcE2E3_fe_correction_columns():
    """Fe correction path produces an E2E3Corrected column."""
    df = _base_df([250, 364, 366], [0.5, 0.2, 0.3])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["Fe"],
        "analyteConcentration": [1.0],
    })

    result = calcE2E3(df, concentrationData=concentration, correctFe=True)

    assert "E2E3Corrected" in result.columns


def test_calcE2E3_fe_correction_value():
    """Fe correction produces the correct numerical E2E3 ratio.

    At λ=250, Fe=1:
      correction = (-0.00000044*62500) - (0.00007755*250) + 0.11337671
                 = -0.0275 - 0.0193875 + 0.11337671 = 0.06648921
      corrected_250 = 0.5 - 0.06648921 = 0.43351079

    At λ=364, Fe=1:
      correction = (-0.00000044*132496) - (0.00007755*364) + 0.11337671
                 = -0.05829824 - 0.02282820 + 0.11337671 = 0.02685027  (note: -0.0282282)
      Wait, 0.00007755*364 = 0.0282282
      correction = -0.05829824 - 0.0282282 + 0.11337671 = 0.02685027
      corrected_364 = 0.2 - 0.02685027 = 0.17314973

    At λ=366, Fe=1:
      correction = (-0.00000044*133956) - (0.00007755*366) + 0.11337671
                 = -0.05894064 - 0.0283923 + 0.11337671 = 0.02604377
      corrected_366 = 0.3 - 0.02604377 = 0.27395623

    avg_corrected_365 = (0.17314973 + 0.27395623) / 2 = 0.22355298
    E2E3Corrected = 0.43351079 / 0.22355298 ≈ 1.9393
    """
    df = _base_df([250, 364, 366], [0.5, 0.2, 0.3])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["Fe"],
        "analyteConcentration": [1.0],
    })

    result = calcE2E3(df, concentrationData=concentration, correctFe=True)

    assert result["E2E3Corrected"].iloc[0] == pytest.approx(1.9392287671142183, rel=1e-6)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_calcE2E3_output_columns():
    """Output has exactly the expected columns when correctFe=False."""
    df = _base_df([250, 364, 366], [0.5, 0.2, 0.3])

    result = calcE2E3(df)

    assert list(result.columns) == ["domainID", "siteID", "sampleID", "collectDate", "E2E3"]


def test_calcE2E3_output_columns_with_fe():
    """Output has exactly the expected columns when correctFe=True."""
    df = _base_df([250, 364, 366], [0.5, 0.2, 0.3])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["Fe"],
        "analyteConcentration": [1.0],
    })

    result = calcE2E3(df, concentrationData=concentration, correctFe=True)

    assert list(result.columns) == [
        "domainID", "siteID", "sampleID", "collectDate", "E2E3", "E2E3Corrected"
    ]


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_calcE2E3_empty_data_raises():
    df = pd.DataFrame(columns=["sampleID", "domainID", "siteID", "collectDate",
                                "wavelength", "decadicAbsorbance"])
    with pytest.raises(ValueError, match="No absorbance data"):
        calcE2E3(df)


def test_calcE2E3_correctFe_no_concentration_raises():
    df = _base_df([250, 364, 366], [0.5, 0.2, 0.3])
    with pytest.raises(ValueError, match="No concentration data"):
        calcE2E3(df, correctFe=True)


def test_calcE2E3_correctFe_no_fe_analyte_raises():
    df = _base_df([250, 364, 366], [0.5, 0.2, 0.3])
    concentration = pd.DataFrame({
        "sampleID": ["s1"],
        "analyte": ["DOC"],
        "analyteConcentration": [5.0],
    })
    with pytest.raises(ValueError, match="No Fe concentration data"):
        calcE2E3(df, concentrationData=concentration, correctFe=True)
