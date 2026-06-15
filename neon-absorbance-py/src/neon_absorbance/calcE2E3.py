# Title: calcE2E3
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# Description: Calculate the E2:E3 absorbance ratio (250 nm : 365 nm) from
#   NEON water chemistry scan data. Optionally applies Fe(III) correction.
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007
#
# changelog and author contributions / copyrights
#   Adriana Teruel (2026-06-15)
#     original creation (Python translation of R package)

import pandas as pd

from ._helpers import _select_wavelength, _fe_absorbance


def calcE2E3(
    absorbanceData: pd.DataFrame,
    concentrationData: pd.DataFrame | None = None,
    correctFe: bool = False,
) -> pd.DataFrame:
    """
    Calculate the E2:E3 absorbance ratio (250 nm : 365 nm) from National
    Ecological Observatory Network (NEON) water chemistry data.

    The 250 nm wavelength is selected directly (even). The 365 nm wavelength
    is estimated by averaging the measured absorbances at 364 nm and 366 nm,
    the two nearest even wavelengths.

    Parameters
    ----------
    absorbanceData : pd.DataFrame
        Table of NEON absorbance data (swc_externalLabAbsorbanceScan).
        Required columns: sampleID, domainID, siteID, collectDate,
        wavelength, decadicAbsorbance.
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
        E2E3. If correctFe=True, also includes E2E3Corrected.

    Raises
    ------
    ValueError
        If absorbanceData is empty, or correctFe=True but no Fe
        concentration data is available.

    Examples
    --------
    >>> result = calcE2E3(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ... )

    >>> result = calcE2E3(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ...     correctFe=True,
    ... )
    """
    if absorbanceData is None or absorbanceData.empty:
        raise ValueError("No absorbance data")

    # Select the two wavelengths. _select_wavelength handles 365 (odd) by
    # selecting 364 and 366 and relabelling them as 365 so that the groupby
    # average below produces a single estimated value.
    absorbanceData250 = _select_wavelength(absorbanceData, 250)
    absorbanceData365 = _select_wavelength(absorbanceData, 365)

    combinedData = pd.concat(
        [absorbanceData250, absorbanceData365],
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

    abs250 = (
        combinedData[combinedData["wavelength"] == 250]
        [["sampleID", "absorbance"]]
        .rename(columns={"absorbance": "abs250"})
    )

    abs365 = (
        combinedData[combinedData["wavelength"] == 365]
        [["sampleID", "absorbance"]]
        .rename(columns={"absorbance": "abs365"})
    )

    E2E3 = abs250.merge(abs365, on="sampleID", how="outer")
    E2E3["E2E3"] = E2E3["abs250"] / E2E3["abs365"]

    outputTable = (
        absorbanceAveraged[["domainID", "siteID", "sampleID", "collectDate"]]
        .drop_duplicates()
        .merge(E2E3[["sampleID", "E2E3"]], on="sampleID", how="left")
    )

    if correctFe:
        abs250Corrected = (
            combinedData[combinedData["wavelength"] == 250]
            [["sampleID", "absorbanceCorrected"]]
            .rename(columns={"absorbanceCorrected": "abs250Corrected"})
        )

        abs365Corrected = (
            combinedData[combinedData["wavelength"] == 365]
            [["sampleID", "absorbanceCorrected"]]
            .rename(columns={"absorbanceCorrected": "abs365Corrected"})
        )

        E2E3Corrected = abs250Corrected.merge(
            abs365Corrected,
            on="sampleID",
            how="outer",
        )
        E2E3Corrected["E2E3Corrected"] = (
            E2E3Corrected["abs250Corrected"] / E2E3Corrected["abs365Corrected"]
        )

        outputTable = outputTable.merge(
            E2E3Corrected[["sampleID", "E2E3Corrected"]],
            on="sampleID",
            how="left",
        )

    return outputTable
