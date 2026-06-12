# Title: calcSuva
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

import pandas as pd

from ._helpers import _select_wavelength, _fe_absorbance


def calcSuva(
    absorbanceData: pd.DataFrame,
    concentrationData: pd.DataFrame,
    wavelength: int,
    correctFe: bool = False,
) -> pd.DataFrame:
    """
    Calculate the specific ultra-violet absorbance (SUVA) at a user-specified
    wavelength from National Ecological Observatory Network (NEON) water
    chemistry data.

    SUVA is calculated as:
        SUVA = (absorbance / DOC concentration) * 100
    in units of L / (mg · m).

    Parameters
    ----------
    absorbanceData : pd.DataFrame
        Table of NEON absorbance data (swc_externalLabAbsorbanceScan).
        Required columns: sampleID, domainID, siteID, collectDate,
        wavelength, decadicAbsorbance.
    concentrationData : pd.DataFrame
        Table of NEON concentration data (swc_externalLabDataByAnalyte).
        Must contain a row where analyte == "DOC" for each sample.
        If correctFe=True, must also contain rows where analyte == "Fe".
    wavelength : int
        Wavelength (nm) at which to calculate SUVA. Must be in the
        220–600 nm range. If odd, the two surrounding even wavelengths
        are averaged.
    correctFe : bool, optional
        Whether to apply a correction for the overlapping absorbance of
        Fe(III). Defaults to False. See README for details.

    Returns
    -------
    pd.DataFrame
        Table with columns: domainID, siteID, sampleID, collectDate,
        suva{wavelength}. If correctFe=True, also includes
        suva{wavelength}Corrected.

    Raises
    ------
    ValueError
        If the wavelength is outside the 220–600 nm range, if no absorbance
        data exists at the specified wavelength, if no DOC concentration data
        is available, or if correctFe=True but no Fe concentration data is
        available.

    Examples
    --------
    >>> result = calcSuva(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ...     wavelength=254,
    ... )

    >>> result = calcSuva(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ...     concentrationData=swc_externalLabDataByAnalyte,
    ...     wavelength=254,
    ...     correctFe=True,
    ... )
    """
    if wavelength < 220 or wavelength > 600:
        raise ValueError("Wavelength is outside 220-600 nm range")

    # Select the specified wavelength (handles odd wavelengths automatically)
    absorbanceData = _select_wavelength(absorbanceData, wavelength)

    if absorbanceData is None or absorbanceData.empty:
        raise ValueError("No absorbance data at the specified wavelength")

    # Average replicate scans, grouping only by sampleID since all rows are
    # now at (or relabelled as) the same wavelength
    absorbanceAveraged = (
        absorbanceData
        .groupby("sampleID", as_index=False)
        .agg(
            domainID=("domainID", "first"),
            siteID=("siteID", "first"),
            collectDate=("collectDate", "first"),
            absorbance=("decadicAbsorbance", "mean"),
        )
    )
    absorbanceAveraged["wavelength"] = wavelength

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

    # SUVA = absorbance / DOC * 100  (units: L / mg-m)
    combinedData["suvaX"] = combinedData["absorbance"] / combinedData["DOC"] * 100

    if correctFe:
        combinedData["absorbanceFe"] = _fe_absorbance(
            combinedData["Fe"],
            combinedData["wavelength"],
        )
        combinedData["absorbanceCorrected"] = (
            combinedData["absorbance"] - combinedData["absorbanceFe"]
        )
        combinedData["suvaXCorrected"] = (
            combinedData["absorbanceCorrected"] / combinedData["DOC"] * 100
        )

    # Output column names include the wavelength (e.g. suva254, suva254Corrected)
    suva_col = f"suva{wavelength}"
    suva_col_corrected = f"suva{wavelength}Corrected"

    outputTable = combinedData[
        ["domainID", "siteID", "sampleID", "collectDate", "suvaX"]
    ].rename(columns={"suvaX": suva_col})

    if correctFe:
        outputTable[suva_col_corrected] = combinedData["suvaXCorrected"].values

    return outputTable
