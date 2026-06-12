# Shared helper functions for neon_absorbance
#
# Original author: Robert Hensley <hensley@battelleecology.org>
# Translated to Python by: Adriana Teruel <teruel@battelleecology.org>
#
# License: GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

import pandas as pd


def _select_wavelength(absorbanceData: pd.DataFrame, wavelength: int) -> pd.DataFrame:
    """Select rows matching a wavelength.

    If the wavelength is even, rows at exactly that wavelength are returned.
    If the wavelength is odd, rows at the two surrounding even wavelengths
    (wavelength-1 and wavelength+1) are returned and relabelled as the
    requested wavelength, so that downstream averaging produces a single
    estimated value.
    """
    if wavelength % 2 == 0:
        selected = absorbanceData[absorbanceData["wavelength"] == wavelength].copy()
    else:
        selected = absorbanceData[
            absorbanceData["wavelength"].isin([wavelength - 1, wavelength + 1])
        ].copy()
        selected["wavelength"] = wavelength

    return selected


def _fe_absorbance(fe, wavelength):
    """Estimate Fe(III) absorbance contribution at a given wavelength.

    Parameters
    ----------
    fe : numeric or pd.Series
        Iron concentration in mg/L.
    wavelength : numeric or pd.Series
        Wavelength in nm.

    Returns
    -------
    numeric or pd.Series
        Estimated absorbance contribution from Fe(III).
    """
    return fe * (
        (-0.00000044 * wavelength**2)
        - (0.00007755 * wavelength)
        + 0.11337671
    )
