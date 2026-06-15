# Title: calcSuva350
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# Description: Calculate SUVA at 350 nm from NEON water chemistry scan data.
#   Scan data only — no discrete absorbance fallback. Optionally applies
#   Fe(III) correction.
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007
#
# changelog and author contributions / copyrights
#   Adriana Teruel (2026-06-15)
#     original creation (Python translation of R package)

import pandas as pd

from ._helpers import _fe_absorbance


def calcSuva350(
    absorbanceData: pd.DataFrame,
    concentrationData: pd.DataFrame,
    correctFe: bool = False,
) -> pd.DataFrame:
    """
    Calculate the specific ultra-violet absorbance at 350 nm (SUVA350) from
    National Ecological Observatory Network (NEON) water chemistry data.

    SUVA350 is calculated as:
        SUVA350 = (absorbance at 350 nm / DOC concentration) * 100
    in units of L / (mg · m).

    Unlike calcSuva254 and calcSuva280, this function uses scan data only —
    there is no discrete absorbance fallback for 350 nm.

    Parameters
    ----------
    absorbanceData : pd.DataFrame
        Table of NEON absorbance scan data (swc_externalLabAbsorbanceScan).
        Required columns: sampleID, domainID, siteID, collectDate,
        wavelength, decadicAbsorbance.
    concentrationData : pd.DataFrame
        Table of NEON concentration data (swc_externalLabDataByAnalyte).
        Must contain rows where analyte == "DOC". If correctFe=True, must
        also contain rows where analyte == "Fe".
    correctFe : bool, optional
        Whether to apply a correction for the overlapping absorbance of
        Fe(III). Defaults to False. See README for details.

    Returns
    -------
    pd.DataFrame
        Table with columns: domainID, siteID, sampleID, collectDate,
        suva350. If correctFe=True, also includes suva350Corrected.

    Raises
    ------
    ValueError
        If no absorbance data exists at 350 nm, if no DOC concentration
        data is available, or if correctFe=True but no Fe concentration
        data is available.

    Examples
    --------
    >>> result = calcSuva350(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ... )

    >>> result = calcSuva350(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ...     correctFe=True,
    ... )
    """
    # Filter scan data to 350 nm only
    absorbanceData350 = absorbanceData[absorbanceData["wavelength"] == 350].copy()

    if absorbanceData350.empty:
        raise ValueError("No absorbance data at 350 nm")

    # Average replicate scans
    absorbanceAveraged = (
        absorbanceData350
        .groupby(["sampleID", "wavelength"], as_index=False)
        .agg(
            domainID=("domainID", "first"),
            siteID=("siteID", "first"),
            collectDate=("collectDate", "first"),
            absorbance=("decadicAbsorbance", "mean"),
        )
    )

    # DOC concentration is always required for SUVA
    DOC = (
        concentrationData[concentrationData["analyte"] == "DOC"]
        [["sampleID", "analyteConcentration"]]
        .rename(columns={"analyteConcentration": "DOC"})
    )

    if DOC.empty:
        raise ValueError("No DOC concentration data")

    combinedData = absorbanceAveraged.merge(DOC, on="sampleID", how="inner")

    # Optionally merge Fe concentration for correction
    if correctFe:
        Fe = (
            concentrationData[concentrationData["analyte"] == "Fe"]
            [["sampleID", "analyteConcentration"]]
            .rename(columns={"analyteConcentration": "Fe"})
        )

        if Fe.empty:
            raise ValueError("No Fe concentration data")

        combinedData = combinedData.merge(Fe, on="sampleID", how="inner")

    # SUVA350 = absorbance / DOC * 100  (units: L / mg-m)
    combinedData["suva350"] = combinedData["absorbance"] / combinedData["DOC"] * 100

    if correctFe:
        combinedData["absorbanceFe"] = _fe_absorbance(
            combinedData["Fe"],
            combinedData["wavelength"],
        )
        combinedData["absorbanceCorrected"] = (
            combinedData["absorbance"] - combinedData["absorbanceFe"]
        )
        combinedData["suva350Corrected"] = (
            combinedData["absorbanceCorrected"] / combinedData["DOC"] * 100
        )

    outputTable = combinedData[
        ["domainID", "siteID", "sampleID", "collectDate", "suva350"]
    ].copy()

    if correctFe:
        outputTable["suva350Corrected"] = combinedData["suva350Corrected"].values

    return outputTable
