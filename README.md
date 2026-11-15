# Agricultural Environmental Data Pipeline

A Python and Streamlit application for collecting, combining, and preparing environmental and climate data for agricultural analysis and machine-learning workflows.

The application takes a geographic location and date range, retrieves data from multiple sources, combines the observations into CSV datasets, and can aggregate the hourly dataset into weekly measurements.

## Features

- Streamlit interface for entering a latitude, longitude, and date range.
- Retrieval of environmental data from NASA POWER, PRISM, and ERA5.
- Retrieval of the monthly North Atlantic Oscillation (NAO) climate index from NOAA's Climate Prediction Center.
- Combination of datasets with different temporal resolutions.
- Separate combined datasets with and without the NAO index.
- Weekly aggregation of both combined datasets.
- Local caching for ERA5 point-and-date requests.
- CSV outputs designed for further analysis and machine-learning preprocessing.

## Data sources

| Source | Temporal resolution | Data used in this project | Spatial meaning |
|---|---|---|---|
| [NASA POWER](https://power.larc.nasa.gov/) | Hourly | `T2M`, `T2MDEW`, `T2MWET`, `RH2M`, `QV2M`, `PS`, `WS10M`, `WD10M`, `U10M`, `V10M`, `PRECTOT`, and `ALLSKY_SFC_SW_DWN` | Values requested for the supplied latitude and longitude |
| [PRISM](https://prism.oregonstate.edu/) | Daily | Minimum, maximum, and mean temperature, and precipitation | Daily gridded observations; PRISM coverage is limited to the contiguous United States for the product used here |
| ERA5 / ECMWF ARCO | Hourly | 2 m temperature and dew point, 10 m wind components, surface pressure, total precipitation, and solar radiation | Nearest available ERA5 grid point to the supplied coordinates; the underlying grid is approximately 0.25° |
| [NOAA CPC NAO](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii) | Monthly | North Atlantic Oscillation index | Large-scale climate index; it is not specific to the selected farm location |

### Temporal alignment

NASA POWER and ERA5 provide hourly observations. Their timestamps are normalized to UTC before they are combined. PRISM provides daily values, which are associated with hourly records by date. NOAA's NAO index is monthly and is matched to environmental observations by calendar month.

The NAO index is therefore repeated across the hourly observations belonging to the same month. It should be interpreted as a monthly climate indicator, not as an hourly measurement.

The current pipeline joins PRISM values using the UTC calendar date. Before interpreting daily PRISM values alongside hourly observations, verify that the daily period represented by the PRISM date label matches the intended calendar-day convention. Similarly, ERA5 accumulated variables such as precipitation and solar radiation may refer to an accumulation interval ending at the validity timestamp, whereas other hourly products may label the start of an interval. Exact timestamp equality does not by itself guarantee that two accumulation periods cover the same hour. These interval conventions should be checked before training or comparing variables across sources.

## Project structure

```text
NASASPACEAPPS/
│
├── main.py
├── pipeline.py
├── weekly_aggregation.py
├── requirements.txt
│
├── APIs/
│   ├── NASApowerAPI.py
│   ├── PRISMAPI.py
│   ├── ERA5API.py
│   └── NOAAAPI.py
│
└── data/
    ├── combined_dataset.csv
    ├── combined_dataset_with_nao.csv
    ├── weekly_dataset.csv
    ├── weekly_dataset_with_nao.csv
    ├── era5_cache/
    └── prism_cache/
```

The `data/` files and cache directories are generated when the application and aggregation script run; they do not need to be created manually.

## Requirements

- Python 3
- Internet access
- A Copernicus Climate Data Store personal access token for the ERA5 ARCO service used by `ERA5API.py`

Install the project's Python dependencies from the project root:

```powershell
python -m pip install streamlit requests pandas numpy xarray zarr dask fsspec openpyxl
```

If a `requirements.txt` file is maintained for the project, install from it instead:

```powershell
python -m pip install -r requirements.txt
```

## ERA5 access configuration

The current `ERA5API.py` reads a Copernicus Climate Data Store personal access token from one of these locations:

1. The `CDSAPI_KEY` environment variable.
2. A file specified by `CDSAPI_RC`.
3. `.cdsapirc` in the project root.
4. `.cdsapirc` in the user's home directory.

For PowerShell, set the token for the current terminal session with:

```powershell
$env:CDSAPI_KEY = "YOUR_CDS_PERSONAL_ACCESS_TOKEN"
```

Alternatively, the code can read a `key:` entry from a `.cdsapirc` file. Keep the token private. Add `.cdsapirc` and any local `.env` file to `.gitignore`; never commit credentials to the repository.

The ERA5 implementation accesses ECMWF's remote geo-chunked Zarr store and selects the nearest grid point for the requested location. It also stores retrieved point/time-range results in `data/era5_cache/` so identical requests can reuse locally cached data.

## Running the application

From the project root, start the Streamlit application with:

```powershell
python -m streamlit run main.py
```

In the application, enter the requested latitude, longitude, date range, and sampling interval, then run the data-generation workflow.

The pipeline writes the combined CSV files into the `data/` directory.

## Generated combined datasets

| File | Contents |
|---|---|
| `data/combined_dataset.csv` | NASA POWER, ERA5, and PRISM data, without the NAO feature |
| `data/combined_dataset_with_nao.csv` | The same environmental data with the NAO feature added for months with valid NAO observations |

The dataset without NAO retains the available environmental observations across the requested date range. The dataset with NAO only retains observations that can be matched to a valid monthly NAO value. As a result, the two files may contain different numbers of rows.

## Weekly aggregation

After the two combined CSV files have been generated, create weekly datasets by running:

```powershell
python weekly_aggregation.py
```

The script reads the two existing combined datasets from `data/` and creates:

| File | Contents |
|---|---|
| `data/weekly_dataset.csv` | Weekly environmental measurements without NAO |
| `data/weekly_dataset_with_nao.csv` | Weekly environmental measurements with NAO |

### Weekly aggregation rules

Weeks run from Monday 00:00 UTC through Sunday 23:59:59 UTC. Partial weeks at the beginning and end of the requested date range are retained.

The aggregation method depends on the variable:

- Most meteorological state variables, such as temperature, humidity, and pressure, are averaged over the week.
- Precipitation and accumulated solar-energy variables are summed.
- Wind direction is aggregated using a circular mean, so compass directions near 0° and 360° are handled correctly.
- PRISM daily values are first reduced to one value per calendar day, avoiding counting the same daily value once for every hourly record. Daily temperature values are averaged across the week; daily precipitation is summed.
- The monthly NAO value is first represented once per calendar day and then averaged over the days represented in each week. A week crossing a month boundary can therefore include values from both months.

Weekly files include `Week_Start`, `Week_End`, `Observation_Count`, and `Days_With_Observations`. The NAO weekly file also includes `NAO_Days_Available`.

**Use an hourly sampling interval (`interval=1`) when generating the source datasets for weekly aggregation.** If the original data have already been sampled every several hours, accumulated precipitation and solar-energy totals may be underestimated. The aggregation script warns when it detects timestamps that are typically more than one hour apart.

## Missing data and limitations

- The pipeline does not automatically impute missing environmental measurements. Missing values should be inspected and handled as part of model preprocessing.
- PRISM does not cover all geographic locations. The selected PRISM product is limited to the contiguous United States; locations outside its coverage may have no PRISM values. A retrieval failure can also be caused by a temporary service or network problem.
- ERA5 point extraction selects the nearest grid point; it does not create a measurement at the exact entered coordinates or interpolate between grid cells.
- NAO is a monthly, large-scale climate index and is not a local measurement. Its value is shared across locations for the same month.
- The dataset with NAO can have fewer rows than the dataset without NAO because observations without a valid NAO match are excluded from that version.
- Weekly aggregation preserves partial weeks at the selected date-range boundaries. These weeks may contain fewer observations or days than a complete Monday-to-Sunday week.
- The weekly aggregation script does not fill missing hours or impute missing variable values.

## Notes for machine-learning use

The CSV files are data-preparation outputs, not automatically validated training sets. Before training a model:

1. Inspect missing values and the number of observations in each week.
2. Decide how to handle missing values in environmental variables.
3. Check that weekly totals are based on a sufficiently complete set of source observations.
4. Use a chronological train/validation/test split when forecasting future conditions to reduce temporal leakage.
5. Keep the dataset with NAO and the dataset without NAO separate when comparing the contribution of the NAO feature.

XGBoost can use data at different temporal resolutions; weekly aggregation is a modelling choice made in this project to produce weekly features and reduce the number of records. It is not a requirement imposed by XGBoost.

## Typical workflow

1. Install the dependencies.
2. Configure the ERA5 access token.
3. Run the Streamlit application and generate the combined datasets.
4. Check the output files in `data/`.
5. Run `python weekly_aggregation.py`.
6. Inspect the two weekly CSV files before using them in model training.

## Data handling

Do not commit API credentials, personal access tokens, local cache files, or large generated datasets to a public repository unless that is intentional. Consider adding the following entries to `.gitignore`:

```gitignore
.cdsapirc
.env
.venv/
__pycache__/
data/era5_cache/
data/prism_cache/
data/*.csv
```

If the generated CSV files are intended to be version-controlled or shared, remove the relevant `data/*.csv` ignore rule and manage their size and provenance deliberately.
