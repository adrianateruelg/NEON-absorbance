import pytest
import pandas as pd
from neon_absorbance import calcSuva350


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abs_df(wavelengths, absorbances, sample_id="s1"):
    return pd.DataFrame({
        "sampleID":          [sample_id] * len(wavelengths),
        "domainID":          ["D01"] * len(wavelengths),
        "siteID":            ["SITE"] * len(wavelengths),
        "collectDate":       ["2024-01-01"] * len(wavelengths),
        "wavelength":        wavelengths,
        "decadicAbsorbance": absorbances,
    })


def _conc_df(doc=5.0, fe=None, sample_id="s1"):
    rows = [{"sampleID": sample_id, "analyte": "DOC",
             "analyteConcentration": doc}]
    if fe is not None:
        rows.append({"sampleID": sample_id, "analyte": "Fe",
                     "analyteConcentration": fe})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core SUVA350 tests
# ---------------------------------------------------------------------------

def test_calcSuva350_basic():
    """suva350 = absorbance / DOC * 100 = 0.5 / 5.0 * 100 = 10.0"""
    df = _abs_df([350], [0.5])
    result = calcSuva350(df, _conc_df(doc=5.0))
    assert result["suva350"].iloc[0] == pytest.approx(10.0)


def test_calcSuva350_replicate_scans_averaged():
    """Duplicate scans are averaged before dividing by DOC."""
    df = _abs_df([350, 350], [0.4, 0.6])
    result = calcSuva350(df, _conc_df(doc=5.0))
    assert result["suva350"].iloc[0] == pytest.approx(10.0)


def test_calcSuva350_ignores_other_wavelengths():
    """Only 350 nm rows are used from absorbanceData."""
    df = _abs_df([254, 350, 365], [0.9, 0.5, 0.3])
    result = calcSuva350(df, _conc_df(doc=5.0))
    assert result["suva350"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Fe correction tests
# ---------------------------------------------------------------------------

def test_calcSuva350_fe_correction_columns():
    df = _abs_df([350], [0.5])
    result = calcSuva350(df, _conc_df(doc=5.0, fe=1.0), correctFe=True)
    assert "suva350Corrected" in result.columns


def test_calcSuva350_fe_correction_value():
    """suva350Corrected = (absorbance - absorbanceFe) / DOC * 100."""
    df = _abs_df([350], [0.5])
    result = calcSuva350(df, _conc_df(doc=5.0, fe=1.0), correctFe=True)

    from neon_absorbance._helpers import _fe_absorbance
    expected = (0.5 - _fe_absorbance(1.0, 350)) / 5.0 * 100

    assert result["suva350Corrected"].iloc[0] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_calcSuva350_output_columns():
    df = _abs_df([350], [0.5])
    result = calcSuva350(df, _conc_df())
    assert list(result.columns) == ["domainID", "siteID", "sampleID", "collectDate", "suva350"]


def test_calcSuva350_output_columns_with_fe():
    df = _abs_df([350], [0.5])
    result = calcSuva350(df, _conc_df(doc=5.0, fe=1.0), correctFe=True)
    assert list(result.columns) == [
        "domainID", "siteID", "sampleID", "collectDate", "suva350", "suva350Corrected"
    ]


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_calcSuva350_no_absorbance_raises():
    df = _abs_df([254], [0.5])   # no 350 nm rows
    with pytest.raises(ValueError, match="No absorbance data"):
        calcSuva350(df, _conc_df())


def test_calcSuva350_no_doc_raises():
    df = _abs_df([350], [0.5])
    conc = pd.DataFrame({
        "sampleID": ["s1"], "analyte": ["Fe"], "analyteConcentration": [1.0],
    })
    with pytest.raises(ValueError, match="No DOC"):
        calcSuva350(df, conc)


def test_calcSuva350_correctFe_no_fe_raises():
    df = _abs_df([350], [0.5])
    with pytest.raises(ValueError, match="No Fe"):
        calcSuva350(df, _conc_df(), correctFe=True)
