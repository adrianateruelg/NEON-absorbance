import pytest
import pandas as pd
from neon_absorbance import calcSuva254


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
        rows.append({"sampleID": sample_id, "analyte": "UV Absorbance (254 nm)",
                     "analyteConcentration": discrete_abs, "domainID": "D01",
                     "siteID": "SITE", "collectDate": "2024-01-01"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core SUVA254 tests
# ---------------------------------------------------------------------------

def test_calcSuva254_basic():
    """suva254 = absorbance / DOC * 100 = 0.5 / 5.0 * 100 = 10.0"""
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0)

    result = calcSuva254(df, conc)

    assert result["suva254"].iloc[0] == pytest.approx(10.0)


def test_calcSuva254_replicate_scans_averaged():
    """Duplicate scans are averaged before dividing by DOC."""
    df = _abs_df([254, 254], [0.4, 0.6])  # mean = 0.5
    conc = _conc_df(doc=5.0)

    result = calcSuva254(df, conc)

    assert result["suva254"].iloc[0] == pytest.approx(10.0)


def test_calcSuva254_ignores_other_wavelengths():
    """Only 254 nm rows are used from absorbanceData."""
    df = _abs_df([254, 300, 365], [0.5, 0.9, 0.3])
    conc = _conc_df(doc=5.0)

    result = calcSuva254(df, conc)

    assert result["suva254"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Discrete absorbance fallback tests
# ---------------------------------------------------------------------------

def test_calcSuva254_uses_discrete_when_no_scan():
    """Falls back to discrete absorbance value when no scan data for a sample."""
    df = _abs_df([254], [0.5], sample_id="s1")           # scan data for s1
    conc = pd.DataFrame([
        {"sampleID": "s1", "analyte": "DOC",
         "analyteConcentration": 5.0, "domainID": "D01",
         "siteID": "SITE", "collectDate": "2024-01-01"},
        {"sampleID": "s2", "analyte": "DOC",
         "analyteConcentration": 4.0, "domainID": "D01",
         "siteID": "SITE", "collectDate": "2024-01-01"},
        {"sampleID": "s2", "analyte": "UV Absorbance (254 nm)",
         "analyteConcentration": 0.4, "domainID": "D01",
         "siteID": "SITE", "collectDate": "2024-01-01"},
    ])

    result = calcSuva254(df, conc).set_index("sampleID")

    assert result.loc["s1", "suva254"] == pytest.approx(10.0)   # 0.5/5.0*100
    assert result.loc["s2", "suva254"] == pytest.approx(10.0)   # 0.4/4.0*100


def test_calcSuva254_scan_takes_priority_over_discrete():
    """Scan data is used over discrete value for the same sample."""
    df = _abs_df([254], [0.5])                    # scan: 0.5
    conc = _conc_df(doc=5.0, discrete_abs=0.9)    # discrete: 0.9 (should be ignored)

    result = calcSuva254(df, conc)

    # Should use scan value 0.5, not discrete 0.9
    assert result["suva254"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Fe correction tests
# ---------------------------------------------------------------------------

def test_calcSuva254_fe_correction_columns():
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)

    result = calcSuva254(df, conc, correctFe=True)

    assert "suva254Corrected" in result.columns


def test_calcSuva254_fe_correction_value():
    """suva254Corrected = (absorbance - absorbanceFe) / DOC * 100."""
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)

    result = calcSuva254(df, conc, correctFe=True)

    from neon_absorbance._helpers import _fe_absorbance
    expected = (0.5 - _fe_absorbance(1.0, 254)) / 5.0 * 100

    assert result["suva254Corrected"].iloc[0] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_calcSuva254_output_columns():
    df = _abs_df([254], [0.5])
    result = calcSuva254(df, _conc_df())
    assert list(result.columns) == ["domainID", "siteID", "sampleID", "collectDate", "suva254"]


def test_calcSuva254_output_columns_with_fe():
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)
    result = calcSuva254(df, conc, correctFe=True)
    assert list(result.columns) == [
        "domainID", "siteID", "sampleID", "collectDate", "suva254", "suva254Corrected"
    ]


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_calcSuva254_no_absorbance_raises():
    """No scan data and no discrete data → ValueError."""
    df = _abs_df([300], [0.3])   # no 254 nm rows
    conc = _conc_df(doc=5.0)

    with pytest.raises(ValueError, match="No absorbance data"):
        calcSuva254(df, conc)


def test_calcSuva254_no_doc_raises():
    df = _abs_df([254], [0.5])
    conc = pd.DataFrame({
        "sampleID": ["s1"], "analyte": ["Fe"],
        "analyteConcentration": [1.0], "domainID": ["D01"],
        "siteID": ["SITE"], "collectDate": ["2024-01-01"],
    })
    with pytest.raises(ValueError, match="No DOC"):
        calcSuva254(df, conc)


def test_calcSuva254_correctFe_no_fe_raises():
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0)
    with pytest.raises(ValueError, match="No Fe"):
        calcSuva254(df, conc, correctFe=True)
