import pytest
import pandas as pd
from neon_absorbance import calcSuva


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
    rows = [{"sampleID": sample_id, "analyte": "DOC", "analyteConcentration": doc}]
    if fe is not None:
        rows.append({"sampleID": sample_id, "analyte": "Fe", "analyteConcentration": fe})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core SUVA tests
# ---------------------------------------------------------------------------

def test_calcSuva_even_wavelength():
    """SUVA254 = absorbance / DOC * 100 = 0.5 / 5.0 * 100 = 10.0"""
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0)

    result = calcSuva(df, conc, wavelength=254)

    assert result["suva254"].iloc[0] == pytest.approx(10.0)


def test_calcSuva_odd_wavelength():
    """Odd wavelength: average neighbours 254 and 256, label as 255."""
    df = _abs_df([254, 256], [0.4, 0.6])   # mean absorbance = 0.5
    conc = _conc_df(doc=5.0)

    result = calcSuva(df, conc, wavelength=255)

    # mean(0.4, 0.6) / 5.0 * 100 = 10.0
    assert result["suva255"].iloc[0] == pytest.approx(10.0)


def test_calcSuva_output_column_named_by_wavelength():
    """Output column is named suva{wavelength}."""
    df = _abs_df([300], [0.3])
    conc = _conc_df(doc=3.0)

    result = calcSuva(df, conc, wavelength=300)

    assert "suva300" in result.columns


def test_calcSuva_replicate_scans_averaged():
    """Duplicate scans are averaged before dividing by DOC."""
    df = _abs_df([254, 254], [0.4, 0.6])   # mean = 0.5
    conc = _conc_df(doc=5.0)

    result = calcSuva(df, conc, wavelength=254)

    assert result["suva254"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Fe correction tests
# ---------------------------------------------------------------------------

def test_calcSuva_fe_correction_columns():
    """Fe correction path produces suva{wavelength}Corrected column."""
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)

    result = calcSuva(df, conc, wavelength=254, correctFe=True)

    assert "suva254Corrected" in result.columns


def test_calcSuva_fe_correction_value():
    """suva254Corrected = (absorbance - absorbanceFe) / DOC * 100.

    At λ=254, Fe=1.0:
      absorbanceFe = (-0.00000044*254^2) - (0.00007755*254) + 0.11337671
                   = (-0.00000044*64516) - 0.0197177 + 0.11337671
                   = -0.02838704 - 0.0197177 + 0.11337671 = 0.06527197
      absorbanceCorrected = 0.5 - 0.06527197 = 0.43472803
      suva254Corrected = 0.43472803 / 5.0 * 100 = 8.6945606
    """
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)

    result = calcSuva(df, conc, wavelength=254, correctFe=True)

    from neon_absorbance._helpers import _fe_absorbance
    fe_correction = _fe_absorbance(1.0, 254)
    expected = (0.5 - fe_correction) / 5.0 * 100

    assert result["suva254Corrected"].iloc[0] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_calcSuva_output_columns():
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0)
    result = calcSuva(df, conc, wavelength=254)
    assert list(result.columns) == ["domainID", "siteID", "sampleID", "collectDate", "suva254"]


def test_calcSuva_output_columns_with_fe():
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0, fe=1.0)
    result = calcSuva(df, conc, wavelength=254, correctFe=True)
    assert list(result.columns) == [
        "domainID", "siteID", "sampleID", "collectDate", "suva254", "suva254Corrected"
    ]


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_calcSuva_wavelength_too_low_raises():
    df = _abs_df([254], [0.5])
    with pytest.raises(ValueError, match="Wavelength"):
        calcSuva(df, _conc_df(), wavelength=100)


def test_calcSuva_wavelength_too_high_raises():
    df = _abs_df([254], [0.5])
    with pytest.raises(ValueError, match="Wavelength"):
        calcSuva(df, _conc_df(), wavelength=700)


def test_calcSuva_no_absorbance_at_wavelength_raises():
    df = _abs_df([300], [0.3])   # no data at 254
    with pytest.raises(ValueError, match="No absorbance data"):
        calcSuva(df, _conc_df(), wavelength=254)


def test_calcSuva_no_doc_raises():
    df = _abs_df([254], [0.5])
    conc = pd.DataFrame({
        "sampleID": ["s1"], "analyte": ["Fe"], "analyteConcentration": [1.0],
    })
    with pytest.raises(ValueError, match="No DOC"):
        calcSuva(df, conc, wavelength=254)


def test_calcSuva_correctFe_no_fe_raises():
    df = _abs_df([254], [0.5])
    conc = _conc_df(doc=5.0)   # no Fe row
    with pytest.raises(ValueError, match="No Fe"):
        calcSuva(df, conc, wavelength=254, correctFe=True)
