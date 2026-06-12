# Title: calcSuva280
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

import pandas as pd

from ._helpers import _fe_absorbance


def calcSuva280(
    absorbanceData: pd.DataFrame,
    concentrationData: pd.DataFrame,
    correctFe: bool = False,
) -> pd.DataFrame:
    """
    Calculate the specific ultra-violet absorbance at 280 nm (SUVA280) from
    National Ecological Observatory Network (NEON) water chemistry data.

    SUVA280 is calculated as:
        SUVA280 = (absorbance at 280 nm / DOC concentration) * 100
    in units of L / (mg · m).

    This function uses full-spectrum scan absorbance data where available.
    For samples without scan data, it falls back to older discrete absorbance
    values stored in concentrationData under analyte == "UV Absorbance (280 nm)".
    Scan data always takes priority over discrete values for the same sample.

    Parameters
    ----------
    absorbanceData : pd.DataFrame
        Table of NEON absorbance scan data (swc_externalLabAbsorbanceScan).
        Required columns: sampleID, domainID, siteID, collectDate,
        wavelength, decadicAbsorbance.
    concentrationData : pd.DataFrame
        Table of NEON concentration data (swc_externalLabDataByAnalyte).
        Must contain rows where analyte == "DOC". May also contain rows
        where analyte == "UV Absorbance (280 nm)" for older discrete
        absorbance values. If correctFe=True, must also contain rows
        where analyte == "Fe".
    correctFe : bool, optional
        Whether to apply a correction for the overlapping absorbance of
        Fe(III). Defaults to False. See README for details.

    Returns
    -------
    pd.DataFrame
        Table with columns: domainID, siteID, sampleID, collectDate,
        suva280. If correctFe=True, also includes suva280Corrected.

    Raises
    ------
    ValueError
        If no absorbance data is available at 280 nm (from either scan or
        discrete sources), if no DOC concentration data is available, or if
        correctFe=True but no Fe concentration data is available.

    Examples
    --------
    >>> result = calcSuva280(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ... )

    >>> result = calcSuva280(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ...     correctFe=True,
    ... )
    """
    # Filter scan data to 280 nm only
    absorbanceData280 = absorbanceData[absorbanceData["wavelength"] == 280].copy()

    # Average replicate scans
    absorbanceAveraged = (
        absorbanceData280
        .groupby(["sampleID", "wavelength"], as_index=False)
        .agg(
            domainID=("domainID", "first"),
            siteID=("siteID", "first"),
            collectDate=("collectDate", "first"),
            absorbance=("decadicAbsorbance", "mean"),
        )
    )

    # Pull discrete absorbance values (older format stored in concentration table)
    absorbanceDiscrete = (
        concentrationData[concentrationData["analyte"] == "UV Absorbance (280 nm)"]
        [["sampleID", "domainID", "siteID", "collectDate", "analyteConcentration"]]
        .copy()
        .rename(columns={"analyteConcentration": "absorbance"})
    )
    absorbanceDiscrete["wavelength"] = 280

    # Only use discrete values for samples not already in the scan data
    absorbanceDiscrete = absorbanceDiscrete[
        ~absorbanceDiscrete["sampleID"].isin(absorbanceAveraged["sampleID"])
    ]

    # Combine scan and discrete absorbance, scan data first
    absorbanceAveraged = pd.concat(
        [absorbanceDiscrete, absorbanceAveraged],
        ignore_index=True,
    )

    if absorbanceAveraged.empty:
        raise ValueError("No absorbance data at 280 nm")

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

    # SUVA280 = absorbance / DOC * 100  (units: L / mg-m)
    combinedData["suva280"] = combinedData["absorbance"] / combinedData["DOC"] * 100

    if correctFe:
        combinedData["absorbanceFe"] = _fe_absorbance(
            combinedData["Fe"],
            combinedData["wavelength"],
        )
        combinedData["absorbanceCorrected"] = (
            combinedData["absorbance"] - combinedData["absorbanceFe"]
        )
        combinedData["suva280Corrected"] = (
            combinedData["absorbanceCorrected"] / combinedData["DOC"] * 100
        )

    outputTable = combinedData[
        ["domainID", "siteID", "sampleID", "collectDate", "suva280"]
    ].copy()

    if correctFe:
        outputTable["suva280Corrected"] = combinedData["suva280Corrected"].values

    return outputTable
