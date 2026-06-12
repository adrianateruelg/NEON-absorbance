# Title: calcSR
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

import numpy as np
import pandas as pd

from ._helpers import _fe_absorbance


def _spectral_slope(group: pd.DataFrame, absorbance_col: str = "absorbance") -> float:
    """Fit log(absorbance) ~ wavelength and return the negated slope.

    The raw regression slope is negative (absorbance decreases with wavelength),
    so we negate it to return a positive spectral slope value, matching the
    convention used in the original R package.
    """
    x = group["wavelength"].values.astype(float)
    y = np.log(group[absorbance_col].values.astype(float))
    slope = np.polyfit(x, y, 1)[0]
    return -slope


def calcSR(
    absorbanceData: pd.DataFrame,
    concentrationData: pd.DataFrame | None = None,
    correctFe: bool = False,
) -> pd.DataFrame:
    """
    Calculate the spectral slope ratio (SR) from National Ecological
    Observatory Network (NEON) water chemistry data.

    SR is defined as the spectral slope over 275–295 nm divided by the
    spectral slope over 350–400 nm. Each slope is the absolute value of
    the coefficient from a linear regression of log(absorbance) on
    wavelength over the respective band.

    Parameters
    ----------
    absorbanceData : pd.DataFrame
        Table of NEON absorbance data (swc_externalLabAbsorbanceScan).
        Required columns: sampleID, domainID, siteID, collectDate,
        wavelength, decadicAbsorbance. Must contain even wavelengths
        spanning 275–295 nm and 350–400 nm.
    concentrationData : pd.DataFrame, optional
        Table of NEON concentration data (swc_externalLabDataByAnalyte).
        Required when correctFe=True. Must contain columns: sampleID,
        analyte, analyteConcentration.
    correctFe : bool, optional
        Whether to apply a correction for the overlapping absorbance of
        Fe(III) before fitting the slopes. Defaults to False.
        See README for details.

    Returns
    -------
    pd.DataFrame
        Table with columns: domainID, siteID, sampleID, collectDate,
        SR. If correctFe=True, also includes SRCorrected.

    Raises
    ------
    ValueError
        If absorbanceData is empty, or correctFe=True but no Fe
        concentration data is available.

    Examples
    --------
    >>> result = calcSR(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ... )

    >>> result = calcSR(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ...     correctFe=True,
    ... )
    """
    if absorbanceData is None or absorbanceData.empty:
        raise ValueError("No absorbance data")

    # Average replicate scans across all wavelengths
    absorbanceAveraged = (
        absorbanceData
        .groupby(["sampleID", "wavelength"], as_index=False)
        .agg(
            domainID=("domainID", "first"),
            siteID=("siteID", "first"),
            collectDate=("collectDate", "first"),
            absorbance=("decadicAbsorbance", "mean"),
        )
    )

    if correctFe:
        if concentrationData is None or concentrationData.empty:
            raise ValueError("No concentration data")

        Fe = (
            concentrationData[concentrationData["analyte"] == "Fe"]
            [["sampleID", "analyteConcentration"]]
            .rename(columns={"analyteConcentration": "Fe"})
        )

        if Fe.empty:
            raise ValueError("No Fe concentration data")

        combinedData = absorbanceAveraged.merge(Fe, on="sampleID", how="inner")
        combinedData = combinedData.dropna()
        combinedData["absorbanceFe"] = _fe_absorbance(
            combinedData["Fe"],
            combinedData["wavelength"],
        )
        combinedData["absorbanceCorrected"] = (
            combinedData["absorbance"] - combinedData["absorbanceFe"]
        )

    # ── Uncorrected slopes ────────────────────────────────────────────────

    band275 = absorbanceAveraged[
        absorbanceAveraged["wavelength"].between(275, 295)
    ]
    band350 = absorbanceAveraged[
        absorbanceAveraged["wavelength"].between(350, 400)
    ]

    slopes275 = (
        band275
        .groupby("sampleID")[["wavelength", "absorbance"]]
        .apply(_spectral_slope)
        .reset_index(name="slope275")
    )

    slopes350 = (
        band350
        .groupby("sampleID")[["wavelength", "absorbance"]]
        .apply(_spectral_slope)
        .reset_index(name="slope350")
    )

    SlopeRatios = slopes275.merge(slopes350, on="sampleID")
    SlopeRatios["SR"] = SlopeRatios["slope275"] / SlopeRatios["slope350"]

    outputTable = (
        absorbanceAveraged[["domainID", "siteID", "sampleID", "collectDate"]]
        .drop_duplicates()
        .merge(SlopeRatios[["sampleID", "SR"]], on="sampleID", how="left")
    )

    # ── Fe-corrected slopes ───────────────────────────────────────────────

    if correctFe:
        band275c = combinedData[combinedData["wavelength"].between(275, 295)]
        band350c = combinedData[combinedData["wavelength"].between(350, 400)]

        slopes275c = (
            band275c
            .groupby("sampleID")[["wavelength", "absorbanceCorrected"]]
            .apply(lambda g: _spectral_slope(g, absorbance_col="absorbanceCorrected"))
            .reset_index(name="slope275")
        )

        slopes350c = (
            band350c
            .groupby("sampleID")[["wavelength", "absorbanceCorrected"]]
            .apply(lambda g: _spectral_slope(g, absorbance_col="absorbanceCorrected"))
            .reset_index(name="slope350")
        )

        SlopeRatiosCorrected = slopes275c.merge(slopes350c, on="sampleID")
        SlopeRatiosCorrected["SRCorrected"] = (
            SlopeRatiosCorrected["slope275"] / SlopeRatiosCorrected["slope350"]
        )

        outputTable = outputTable.merge(
            SlopeRatiosCorrected[["sampleID", "SRCorrected"]],
            on="sampleID",
            how="left",
        )

    return outputTable
