# Title: calcAbsRatio
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# Description: Calculate the absorbance ratio for two user-specified wavelengths
#   from NEON water chemistry scan data. Optionally applies Fe(III) correction.
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007
#
# changelog and author contributions / copyrights
#   Adriana Teruel (2026-06-15)
#     original creation (Python translation of R package)

import pandas as pd

from ._helpers import _select_wavelength, _fe_absorbance


def calcAbsRatio(
    absorbanceData: pd.DataFrame,
    wavelength1: int,
    wavelength2: int,
    concentrationData: pd.DataFrame | None = None,
    correctFe: bool = False,
) -> pd.DataFrame:
    """
    Calculate the absorbance ratio for two user-specified wavelengths from
    National Ecological Observatory Network (NEON) water chemistry data.

    Parameters
    ----------
    absorbanceData : pd.DataFrame
        Table of NEON absorbance data (swc_externalLabAbsorbanceScan).
        Required columns: sampleID, domainID, siteID, collectDate,
        wavelength, decadicAbsorbance.
    wavelength1 : int
        Wavelength (nm) used as the numerator of the ratio. Must be in
        the 220–600 nm range. If odd, the two surrounding even wavelengths
        are averaged.
    wavelength2 : int
        Wavelength (nm) used as the denominator of the ratio. Must be in
        the 220–600 nm range. If odd, the two surrounding even wavelengths
        are averaged.
    concentrationData : pd.DataFrame, optional
        Table of NEON concentration data (swc_externalLabDataByAnalyte).
        Required when correctFe=True. Must contain columns: sampleID,
        analyte, analyteConcentration.
    correctFe : bool, optional
        Whether to apply a correction for the overlapping absorbance of
        Fe(III). Defaults to False. See README for details.

    Returns
    -------
    pd.DataFrame
        Table with columns: domainID, siteID, sampleID, collectDate,
        absRatio. If correctFe=True, also includes absRatioCorrected.

    Raises
    ------
    ValueError
        If absorbanceData is empty, either wavelength is outside 220–600 nm,
        or correctFe=True but no Fe concentration data is available.

    Examples
    --------
    >>> result = calcAbsRatio(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     wavelength1=254,
    ...     wavelength2=365,
    ... )

    >>> result = calcAbsRatio(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     wavelength1=254,
    ...     wavelength2=365,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ...     correctFe=True,
    ... )
    """
    if absorbanceData is None or absorbanceData.empty:
        raise ValueError("No absorbance data")

    if wavelength1 < 220 or wavelength1 > 600:
        raise ValueError("Wavelength 1 is outside 220-600 nm range")

    if wavelength2 < 220 or wavelength2 > 600:
        raise ValueError("Wavelength 2 is outside 220-600 nm range")

    absorbanceData1 = _select_wavelength(absorbanceData, wavelength1)
    absorbanceData2 = _select_wavelength(absorbanceData, wavelength2)

    combinedData = pd.concat(
        [absorbanceData1, absorbanceData2],
        ignore_index=True,
    )

    absorbanceAveraged = (
        combinedData
        .groupby(["sampleID", "wavelength"], as_index=False)
        .agg(
            domainID=("domainID", "first"),
            siteID=("siteID", "first"),
            collectDate=("collectDate", "first"),
            absorbance=("decadicAbsorbance", "mean"),
        )
    )

    combinedData = absorbanceAveraged.copy()

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

        combinedData = combinedData.merge(Fe, on="sampleID", how="inner")
        combinedData["absorbanceFe"] = _fe_absorbance(
            combinedData["Fe"],
            combinedData["wavelength"],
        )
        combinedData["absorbanceCorrected"] = (
            combinedData["absorbance"] - combinedData["absorbanceFe"]
        )

    abs1 = (
        combinedData[combinedData["wavelength"] == wavelength1]
        [["sampleID", "absorbance"]]
        .rename(columns={"absorbance": "abs1"})
    )

    abs2 = (
        combinedData[combinedData["wavelength"] == wavelength2]
        [["sampleID", "absorbance"]]
        .rename(columns={"absorbance": "abs2"})
    )

    absRatio = abs1.merge(abs2, on="sampleID", how="outer")
    absRatio["absRatio"] = absRatio["abs1"] / absRatio["abs2"]

    outputTable = (
        absorbanceAveraged[["domainID", "siteID", "sampleID", "collectDate"]]
        .drop_duplicates()
        .merge(absRatio[["sampleID", "absRatio"]], on="sampleID", how="left")
    )

    if correctFe:
        abs1Corrected = (
            combinedData[combinedData["wavelength"] == wavelength1]
            [["sampleID", "absorbanceCorrected"]]
            .rename(columns={"absorbanceCorrected": "abs1Corrected"})
        )

        abs2Corrected = (
            combinedData[combinedData["wavelength"] == wavelength2]
            [["sampleID", "absorbanceCorrected"]]
            .rename(columns={"absorbanceCorrected": "abs2Corrected"})
        )

        absRatioCorrected = abs1Corrected.merge(
            abs2Corrected,
            on="sampleID",
            how="outer",
        )
        absRatioCorrected["absRatioCorrected"] = (
            absRatioCorrected["abs1Corrected"]
            / absRatioCorrected["abs2Corrected"]
        )

        outputTable = outputTable.merge(
            absRatioCorrected[["sampleID", "absRatioCorrected"]],
            on="sampleID",
            how="left",
        )

    return outputTable
