# Title: formatWide
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# Description: Reformat NEON UV-Vis absorbance data from long format (one row
#   per sample x wavelength) into wide format (one row per sample, one column
#   per wavelength).
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007
#
# changelog and author contributions / copyrights
#   Adriana Teruel (2026-06-15)
#     original creation (Python translation of R package)

import pandas as pd


def formatWide(absorbanceData: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat NEON UV-Vis absorbance data from long format into wide format.

    The input table has one row per sample × wavelength combination. The
    output table has one row per sample, with each wavelength as its own
    column named absorbance.{wavelength} (e.g. absorbance.254, absorbance.365).

    Replicate scans at the same sample × wavelength are averaged before
    pivoting.

    Parameters
    ----------
    absorbanceData : pd.DataFrame
        Table of NEON absorbance scan data (swc_externalLabAbsorbanceScan).
        Required columns: sampleID, domainID, siteID, collectDate,
        wavelength, decadicAbsorbance.

    Returns
    -------
    pd.DataFrame
        Wide-format table with columns: domainID, siteID, sampleID,
        collectDate, absorbance.220, absorbance.222, ..., absorbance.600
        (one column per wavelength present in the input data).

    Raises
    ------
    ValueError
        If absorbanceData is empty.

    Examples
    --------
    >>> result = formatWide(
    ...     absorbanceData=swc_externalLabAbsorbanceScan,
    ... )
    """
    if absorbanceData is None or absorbanceData.empty:
        raise ValueError("No absorbance data")

    # Average replicate scans at each sampleID × wavelength
    absorbanceAveraged = (
        absorbanceData
        .groupby(["sampleID", "wavelength"], as_index=False)
        .agg(absorbance=("decadicAbsorbance", "mean"))
    )

    # Pivot from long to wide: each wavelength becomes its own column
    wideFormat = absorbanceAveraged.pivot(
        index="sampleID",
        columns="wavelength",
        values="absorbance",
    )

    # Name columns to match R output: absorbance.{wavelength}
    wideFormat.columns = [f"absorbance.{int(w)}" for w in wideFormat.columns]
    wideFormat = wideFormat.reset_index()

    # Join sample metadata (domainID, siteID, collectDate) back on
    metadata = (
        absorbanceData[["domainID", "siteID", "sampleID", "collectDate"]]
        .drop_duplicates(subset="sampleID")
    )

    outputTable = metadata.merge(wideFormat, on="sampleID", how="left")

    return outputTable
