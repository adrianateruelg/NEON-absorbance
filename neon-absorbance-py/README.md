# neon-absorbance-py

Python translation of the [NEON absorbance R package](https://github.com/NEONScience/NEON-absorbance).

## Description

This package provides functions for working with NEON water chemistry UV-Vis absorbance data and calculating common metrics. It is designed for use with the NEON chemical properties of surface water data product (DP1.20093.001).

## Installation

```bash
pip install neon-absorbance-py
```

Or install from source:

```bash
git clone https://github.com/<your-org>/neon-absorbance-py.git
cd neon-absorbance-py
pip install -e .
```

## Usage

### `calcAbsRatio`

Calculates the absorbance ratio for two user-specified wavelengths.

```python
from neon_absorbance import calcAbsRatio

# Basic ratio (no Fe correction)
result = calcAbsRatio(
    absorbanceData=swc_externalLabAbsorbanceScan,
    wavelength1=254,
    wavelength2=365,
)

# With Fe(III) correction
result = calcAbsRatio(
    absorbanceData=swc_externalLabAbsorbanceScan,
    wavelength1=254,
    wavelength2=365,
    concentrationData=swc_externalLabDataByAnalyte,
    correctFe=True,
)
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `absorbanceData` | DataFrame | NEON absorbance scan table (`swc_externalLabAbsorbanceScan`) |
| `wavelength1` | int | Numerator wavelength in nm (220–600). If odd, the two surrounding even wavelengths are averaged. |
| `wavelength2` | int | Denominator wavelength in nm (220–600). If odd, the two surrounding even wavelengths are averaged. |
| `concentrationData` | DataFrame | NEON concentration table (`swc_externalLabDataByAnalyte`). Required when `correctFe=True`. |
| `correctFe` | bool | Apply Fe(III) absorbance correction. Defaults to `False`. |

**Returns**

A DataFrame with columns `domainID`, `siteID`, `sampleID`, `collectDate`, and `absRatio`. When `correctFe=True`, also includes `absRatioCorrected`.

### Fe(III) Correction

Fe(III) absorbs UV light and can interfere with absorbance-based metrics such as SUVA254. When `correctFe=True`, the function subtracts the estimated Fe(III) absorbance contribution at each wavelength before computing the ratio. The correction uses the empirical polynomial:

```
absorbanceFe = [Fe] × (−0.00000044 × λ² − 0.00007755 × λ + 0.11337671)
```

where `[Fe]` is the iron concentration in mg/L and `λ` is wavelength in nm. This equation was derived from lab measurements of Fe(III) standards across the 220–600 nm range.

Provide `concentrationData` containing a row where `analyte == "Fe"` for each sample to use this correction.

## Data

Input data should come from the NEON Data Portal under data product **DP1.20093.001** (Chemical properties of surface water):

- `swc_externalLabAbsorbanceScan` — absorbance scans across the UV-Vis spectrum
- `swc_externalLabDataByAnalyte` — analyte concentrations including Fe

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Credits & Acknowledgements

Original R package authored by Robert Hensley (hensley@battelleecology.org), Battelle Ecology.

Translated to Python by Adriana Teruel (teruel@battelleecology.org), Battelle Ecology.

The National Ecological Observatory Network is a project solely funded by the National Science Foundation and managed under cooperative agreement by Battelle. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

## License

GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

## Disclaimer

Information and documents contained within this package are available as-is. Codes or documents, or their use, may not be supported or maintained under any program or service and may not be compatible with data currently available from the NEON Data Portal.
