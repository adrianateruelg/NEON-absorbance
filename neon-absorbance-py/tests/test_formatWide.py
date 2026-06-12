import pytest
import pandas as pd
from neon_absorbance import formatWide


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


# ---------------------------------------------------------------------------
# Core pivot tests
# ---------------------------------------------------------------------------

def test_formatWide_basic_pivot():
    """Each wavelength becomes its own column named absorbance.{wavelength}."""
    df = _abs_df([254, 365], [0.5, 0.2])

    result = formatWide(df)

    assert "absorbance.254" in result.columns
    assert "absorbance.365" in result.columns
    assert result["absorbance.254"].iloc[0] == pytest.approx(0.5)
    assert result["absorbance.365"].iloc[0] == pytest.approx(0.2)


def test_formatWide_one_row_per_sample():
    """Output has one row per sample regardless of how many wavelengths."""
    df = _abs_df([254, 300, 365, 400], [0.5, 0.4, 0.2, 0.1])

    result = formatWide(df)

    assert len(result) == 1


def test_formatWide_multiple_samples():
    """Each sample gets its own row."""
    df = pd.concat([
        _abs_df([254, 365], [0.5, 0.2], sample_id="s1"),
        _abs_df([254, 365], [0.3, 0.1], sample_id="s2"),
    ], ignore_index=True)

    result = formatWide(df).set_index("sampleID")

    assert result.loc["s1", "absorbance.254"] == pytest.approx(0.5)
    assert result.loc["s2", "absorbance.254"] == pytest.approx(0.3)


def test_formatWide_replicate_scans_averaged():
    """Duplicate scans at the same wavelength are averaged before pivoting."""
    df = _abs_df([254, 254, 365], [0.4, 0.6, 0.2])

    result = formatWide(df)

    # mean(0.4, 0.6) = 0.5
    assert result["absorbance.254"].iloc[0] == pytest.approx(0.5)
    assert result["absorbance.365"].iloc[0] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_formatWide_metadata_columns_present():
    """Output includes domainID, siteID, sampleID, collectDate."""
    df = _abs_df([254], [0.5])
    result = formatWide(df)
    for col in ["domainID", "siteID", "sampleID", "collectDate"]:
        assert col in result.columns


def test_formatWide_metadata_columns_first():
    """Metadata columns come before wavelength columns."""
    df = _abs_df([254, 365], [0.5, 0.2])
    result = formatWide(df)
    assert list(result.columns[:4]) == ["domainID", "siteID", "sampleID", "collectDate"]


def test_formatWide_column_naming():
    """Wavelength columns are named absorbance.{int} not absorbance.{float}."""
    df = _abs_df([254, 365], [0.5, 0.2])
    result = formatWide(df)
    assert "absorbance.254" in result.columns
    assert "absorbance.365" in result.columns
    # Should not have float-named columns like absorbance.254.0
    assert "absorbance.254.0" not in result.columns


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------

def test_formatWide_empty_data_raises():
    df = pd.DataFrame(columns=["sampleID", "domainID", "siteID", "collectDate",
                                "wavelength", "decadicAbsorbance"])
    with pytest.raises(ValueError, match="No absorbance data"):
        formatWide(df)
