import pytest
import pandas as pd
from neon_absorbance import calcSuva280


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


def _conc_df(doc=5.0, fe=None, discrete_abs=None, sample_id="s1"):
    rows = [{"sampleID": sample_id, "analyte": "DOC",
             "analyteConcentration": doc, "domainID": "D01",
             "siteID": "SITE", "collectDate": "2024-01-01"}]
    if fe is not None:
        rows.append({"sampleID": sample_id, "analyte": "Fe",
                     "analyteConcentration": fe, "domainID": "D01",
                     "siteID": "SITE", "collectDate": "2024-01-01"})
    if discrete_abs is not None:
        rows.append({"sampleID": sample_id, "analyte": "UV Absorbance (280 nm)",
                     "analyteConcentration": discrete_abs, "domainID": "D01",
                     "siteID": "SITE", "collectDate": "2024-01-01"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core SUVA280 tests
# ---------------------------------------------------------------------------

def test_calcSuva280_basic():
    """suva280 = absorbance / DOC * 100 = 0.5 / 5.0 * 100 = 10.0"""
    df = _abs_df([280], [0.5])
    result = calcSuva280(df, _conc_df(doc=5.0))
    assert result["suva280"].iloc[0] == pytest.approx(10.0)


def test_calcSuva280_replicate_scans_averaged():
    """Duplicate scans are averaged before dividing by DOC."""
    df = _abs_df([280, 280], [0.4, 0.6])
    result = calcSuva280(df, _conc_df(doc=5.0))
    assert result["suva280"].iloc[0] == pytest.approx(10.0)


def test_calcSuva280_ignores_other_wavelengths():
    """Only 280 nm rows are used from absorbanceData."""
    df = _abs_df([254, 280, 365], [0.9, 0.5, 0.3])
    result = calcSuva280(df, _conc_df(doc=5.0))
    assert result["suva280"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Discrete absorbance fallback tests
# ---------------------------------------------------------------------------

def test_calcSuva280_uses_discrete_when_no_scan():
    """Falls back to discrete absorbance value when no scan data for a sample."""
    df = _abs_df([280], [0.5], sample_id="s1")
    conc = pd.DataFrame([
        {"sampleID": "s1", "analyte": "DOC",
         "analyteConcentration": 5.0, "domainID": "D01",
         "siteID": "SITE", "collectDate": "2024-01-01"},
        {"sampleID": "s2", "analyte": "DOC",
         "analyteConcentration": 4.0, "domainID": "D01",
         "siteID": "SITE", "collectDate": "2024-01-01"},
        {"sampleID": "s2", "analyte": "UV Absorbance (280 nm)",
         "analyteConcentration": 0.4, "domainID": "D01",
         "siteID": "SITE", "collectDate": "2024-01-01"},
    ])

    result = calcSuva280(df, conc).set_index("sampleID")

    assert result.loc["s1", "suva280"] == pytest.approx(10.0)
    assert result.loc["s2", "suva280"] == pytest.approx(10.0)


def test_calcSuva280_scan_takes_priority_over_discrete():
    """Scan data is used over discrete value for the same sample."""
    df = _abs_df([280], [0.5])
    conc = _conc_df(doc=5.0, discrete_abs=0.9)
    result = calcSuva280(df, conc)
    assert result["suva280"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Fe correction tests
# ---------------------------------------------------------------------------

def test_calcSuva280_fe_correction_columns():
    df = _abs_df([280], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)
    result = calcSuva280(df, conc, correctFe=True)
    assert "suva280Corrected" in result.columns


def test_calcSuva280_fe_correction_value():
    """suva280Corrected = (absorbance - absorbanceFe) / DOC * 100."""
    df = _abs_df([280], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)
    result = calcSuva280(df, conc, correctFe=True)

    from neon_absorbance._helpers import _fe_absorbance
    expected = (0.5 - _fe_absorbance(1.0, 280)) / 5.0 * 100

    assert result["suva280Corrected"].iloc[0] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_calcSuva280_output_columns():
    df = _abs_df([280], [0.5])
    result = calcSuva280(df, _conc_df())
    assert list(result.columns) == ["domainID", "siteID", "sampleID", "collectDate", "suva280"]


def test_calcSuva280_output_columns_with_fe():
    df = _abs_df([280], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)
    result = calcSuva280(df, conc, correctFe=True)
    assert list(result.columns) == [
        "domainID", "siteID", "sampleID", "collectDate", "suva280", "suva280Corrected"
    ]


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_calcSuva280_no_absorbance_raises():
    df = _abs_df([254], [0.5])   # no 280 nm rows
    with pytest.raises(ValueError, match="No absorbance data"):
        calcSuva280(df, _conc_df())


def test_calcSuva280_no_doc_raises():
    df = _abs_df([280], [0.5])
    conc = pd.DataFrame({
        "sampleID": ["s1"], "analyte": ["Fe"],
        "analyteConcentration": [1.0], "domainID": ["D01"],
        "siteID": ["SITE"], "collectDate": ["2024-01-01"],
    })
    with pytest.raises(ValueError, match="No DOC"):
        calcSuva280(df, conc)


def test_calcSuva280_correctFe_no_fe_raises():
    df = _abs_df([280], [0.5])
    with pytest.raises(ValueError, match="No Fe"):
        calcSuva280(df, _conc_df(), correctFe=True)
