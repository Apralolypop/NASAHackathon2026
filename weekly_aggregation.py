"""Create weekly versions of the two datasets produced by main.py.

Expected inputs in ./data:
    combined_dataset.csv
    combined_dataset_with_nao.csv

Outputs in ./data:
    weekly_dataset.csv
    weekly_dataset_with_nao.csv

Weeks run Monday 00:00 UTC through Sunday 23:59:59 UTC. Partial weeks at
requested date-range boundaries are retained. The script does not impute
missing environmental measurements.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. File configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_ROOT / "data"

INPUT_WITHOUT_NAO = DATA_FOLDER / "combined_dataset.csv"
INPUT_WITH_NAO = DATA_FOLDER / "combined_dataset_with_nao.csv"

OUTPUT_WITHOUT_NAO = DATA_FOLDER / "weekly_dataset.csv"
OUTPUT_WITH_NAO = DATA_FOLDER / "weekly_dataset_with_nao.csv"

TIMESTAMP_COLUMN = "Timestamp"

# These columns represent accumulated quantities over the source interval.
# Their weekly values should be totals rather than averages.
SUM_COLUMNS = {
    "PRECTOT",
    "PRISM_ppt",
    "ERA5_total_precipitation_mm",
    "ALLSKY_SFC_SW_DWN",
    "ERA5_solar_radiation_J_m2",
}

# These are daily PRISM values repeated on each hourly row in the source CSV.
# They must be reduced to one value per date before weekly aggregation.
PRISM_DAILY_COLUMNS = {
    "PRISM_tmin",
    "PRISM_tmax",
    "PRISM_tmean",
    "PRISM_ppt",
}


# ---------------------------------------------------------------------
# 2. Aggregation helpers
# ---------------------------------------------------------------------

def week_start(timestamps):
    """Return the Monday (00:00) starting each timestamp's UTC week."""
    normalized = timestamps.dt.normalize()
    return normalized - pd.to_timedelta(normalized.dt.weekday, unit="D")


def sum_with_missing(series):
    """Sum values while keeping an all-missing group as NaN, not zero."""
    return series.sum(min_count=1)


def circular_mean_degrees(series):
    """Calculate a circular mean for compass directions in degrees."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    radians = np.deg2rad(values.to_numpy(dtype=float))
    mean_sin = np.sin(radians).mean()
    mean_cos = np.cos(radians).mean()

    # An almost-zero resultant vector has no well-defined mean direction.
    if np.hypot(mean_sin, mean_cos) < 1e-12:
        return np.nan

    direction = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)
    return 0.0 if np.isclose(direction, 360.0) else direction


def aggregation_for_column(column):
    """Choose a weekly aggregation based on the variable's meaning."""
    if column in SUM_COLUMNS:
        return sum_with_missing

    upper_name = column.upper()
    if "WIND_DIRECTION" in upper_name or upper_name == "WD10M":
        return circular_mean_degrees

    # Most environmental state variables are represented by their weekly mean.
    return "mean"


def empty_weekly_frame(source_columns, include_nao=False):
    """Build an empty result that still has useful output columns."""
    columns = [
        "Week_Start",
        "Week_End",
        "Observation_Count",
        "Days_With_Observations",
    ]

    for coordinate in ("Latitude", "Longitude"):
        if coordinate in source_columns:
            columns.append(coordinate)

    columns.extend(
        column
        for column in source_columns
        if column not in {TIMESTAMP_COLUMN, "Latitude", "Longitude"}
    )

    if include_nao and "NAO_Days_Available" not in columns:
        columns.append("NAO_Days_Available")

    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


# ---------------------------------------------------------------------
# 3. Aggregate one input CSV to weekly measures
# ---------------------------------------------------------------------

def aggregate_weekly(df, include_nao=False):
    """Aggregate an hourly environmental dataset into Monday-based weeks.

    Hourly variables:
      - Most variables: weekly mean.
      - Precipitation and accumulated solar energy: weekly sum.
      - Wind direction: circular mean.

    PRISM variables:
      - First reduce repeated hourly copies to one value per calendar date.
      - Average daily temperature statistics across the week.
      - Sum daily precipitation across the week.

    NAO:
      - First reduce repeated hourly copies to one value per calendar date.
      - Average the daily-represented monthly index values within each week.
        A week crossing a month boundary is therefore weighted by the number
        of represented days from each month.
    """
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Input dataset must contain a '{TIMESTAMP_COLUMN}' column."
        )

    if df.empty:
        return empty_weekly_frame(df.columns, include_nao=include_nao)

    result = df.copy()

    # The source pipeline stores timestamps as UTC-naive values. Parsing with
    # utc=True makes the UTC convention explicit and handles offset-aware input.
    result[TIMESTAMP_COLUMN] = (
        pd.to_datetime(result[TIMESTAMP_COLUMN], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    result = result.dropna(subset=[TIMESTAMP_COLUMN]).copy()

    if result.empty:
        return empty_weekly_frame(df.columns, include_nao=include_nao)

    # Ensure environmental columns that should be numeric are treated as such.
    for column in result.columns:
        if column != TIMESTAMP_COLUMN:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if include_nao:
        if "NAO_index" not in result.columns:
            raise ValueError(
                "The NAO input file does not contain an 'NAO_index' column."
            )
        # Do not let a missing NAO value enter the weekly NAO dataset.
        result = result.dropna(subset=["NAO_index"]).copy()

        if result.empty:
            return empty_weekly_frame(df.columns, include_nao=True)

    result["_Week_Start"] = week_start(result[TIMESTAMP_COLUMN])

    # Detect data that was already downsampled before weekly aggregation.
    unique_timestamps = (
        result[TIMESTAMP_COLUMN]
        .drop_duplicates()
        .sort_values()
    )
    timestamp_diffs = unique_timestamps.diff().dropna()

    if not timestamp_diffs.empty:
        typical_step = timestamp_diffs.median()
        if typical_step > pd.Timedelta(hours=1):
            print(
                "WARNING: The input timestamps are typically more than "
                "one hour apart. Weekly precipitation/solar totals may be "
                "underestimated if main.py previously sampled every 2, 3, "
                "or more hours. Use interval=1 in main.py for reliable "
                "weekly accumulated totals."
            )

    # ----------------------------------------------
    # Aggregate hourly variables and observation counts
    # ----------------------------------------------

    numeric_columns = result.select_dtypes(include="number").columns.tolist()

    prism_columns = [
        column
        for column in numeric_columns
        if column in PRISM_DAILY_COLUMNS
    ]

    excluded_from_hourly = set(prism_columns)
    if "NAO_index" in numeric_columns:
        excluded_from_hourly.add("NAO_index")

    coordinate_columns = [
        column
        for column in ("Latitude", "Longitude")
        if column in numeric_columns
    ]

    hourly_columns = [
        column
        for column in numeric_columns
        if column not in excluded_from_hourly
        and column not in coordinate_columns
    ]

    grouped = result.groupby("_Week_Start", sort=True)

    summary_aggregations = {
        "Observation_Count": (TIMESTAMP_COLUMN, "size"),
        "Days_With_Observations": (
            TIMESTAMP_COLUMN,
            lambda values: values.dt.normalize().nunique(),
        ),
    }

    for column in coordinate_columns:
        summary_aggregations[column] = (column, "first")

    weekly = grouped.agg(**summary_aggregations)

    if hourly_columns:
        hourly_aggregations = {
            column: aggregation_for_column(column)
            for column in hourly_columns
        }
        hourly_weekly = grouped[hourly_columns].agg(hourly_aggregations)
        weekly = weekly.join(hourly_weekly)

    # ----------------------------------------------
    # Aggregate PRISM once per day, not once per hour
    # ----------------------------------------------

    if prism_columns:
        daily_prism = result[[TIMESTAMP_COLUMN] + prism_columns].copy()
        daily_prism["_Date"] = daily_prism[TIMESTAMP_COLUMN].dt.normalize()

        # PRISM's same daily value is repeated on the hourly rows after the
        # main pipeline merge. Grouping by date prevents hourly over-weighting.
        daily_prism = (
            daily_prism
            .groupby("_Date", as_index=False)[prism_columns]
            .first()
        )

        daily_prism["_Week_Start"] = week_start(daily_prism["_Date"])

        prism_aggregations = {
            column: (
                sum_with_missing if column == "PRISM_ppt" else "mean"
            )
            for column in prism_columns
        }

        prism_weekly = daily_prism.groupby("_Week_Start")[
            prism_columns
        ].agg(prism_aggregations)

        weekly = weekly.join(prism_weekly)

    # ----------------------------------------------
    # Aggregate NAO once per day
    # ----------------------------------------------

    if include_nao:
        daily_nao = result[[TIMESTAMP_COLUMN, "NAO_index"]].copy()
        daily_nao["_Date"] = daily_nao[TIMESTAMP_COLUMN].dt.normalize()

        daily_nao = (
            daily_nao
            .groupby("_Date", as_index=False)["NAO_index"]
            .first()
        )

        daily_nao["_Week_Start"] = week_start(daily_nao["_Date"])

        nao_weekly = daily_nao.groupby("_Week_Start").agg(
            NAO_index=("NAO_index", "mean"),
            NAO_Days_Available=("NAO_index", "count"),
        )

        weekly = weekly.join(nao_weekly)

    # ----------------------------------------------
    # Finalize weekly labels and column order
    # ----------------------------------------------

    weekly = weekly.reset_index().rename(
        columns={"_Week_Start": "Week_Start"}
    )

    weekly["Week_End"] = (
        weekly["Week_Start"] + pd.Timedelta(days=6)
    )

    leading_columns = [
        "Week_Start",
        "Week_End",
        "Observation_Count",
        "Days_With_Observations",
    ]

    if "NAO_Days_Available" in weekly.columns:
        leading_columns.append("NAO_Days_Available")

    leading_columns.extend(
        column
        for column in ("Latitude", "Longitude")
        if column in weekly.columns
    )

    remaining_columns = [
        column for column in weekly.columns
        if column not in leading_columns
    ]

    weekly = weekly[leading_columns + remaining_columns]
    weekly = weekly.sort_values("Week_Start").reset_index(drop=True)

    return weekly


# ---------------------------------------------------------------------
# 4. Read both existing datasets and create weekly CSV files
# ---------------------------------------------------------------------

def generate_weekly_datasets(data_folder=DATA_FOLDER):
    """Create weekly CSV versions of both datasets generated by main.py."""
    data_folder = Path(data_folder)

    input_without_nao = data_folder / INPUT_WITHOUT_NAO.name
    input_with_nao = data_folder / INPUT_WITH_NAO.name

    output_without_nao = data_folder / OUTPUT_WITHOUT_NAO.name
    output_with_nao = data_folder / OUTPUT_WITH_NAO.name

    if not input_without_nao.is_file():
        raise FileNotFoundError(
            f"Could not find the input dataset: {input_without_nao}"
        )

    if not input_with_nao.is_file():
        raise FileNotFoundError(
            f"Could not find the input dataset: {input_with_nao}"
        )

    print(f"Reading: {input_without_nao}")
    original_df = pd.read_csv(input_without_nao)

    print(f"Reading: {input_with_nao}")
    nao_df = pd.read_csv(input_with_nao)

    # ----------------------------------------------
    # Weekly dataset WITHOUT NAO
    # ----------------------------------------------

    print("Aggregating dataset without NAO into weekly measures...")

    weekly_df = aggregate_weekly(
        original_df,
        include_nao=False
    )

    weekly_df.to_csv(
        output_without_nao,
        index=False
    )

    # ----------------------------------------------
    # Weekly dataset WITH NAO
    # ----------------------------------------------

    print("Aggregating dataset with NAO into weekly measures...")

    weekly_nao_df = aggregate_weekly(
        nao_df,
        include_nao=True
    )

    # The input with NAO may omit entire months where NAO is unavailable.
    # A week crossing from a month with NAO into one without it would then
    # contain only part of the week's environmental observations. Exclude such
    # partial-coverage weeks by comparing represented days with the base file.
    if not weekly_nao_df.empty and not weekly_df.empty:
        coverage = weekly_nao_df[
            ["Week_Start", "Days_With_Observations"]
        ].merge(
            weekly_df[["Week_Start", "Days_With_Observations"]],
            on="Week_Start",
            how="left",
            suffixes=("_nao", "_all")
        )

        complete_weeks = coverage.loc[
            coverage["Days_With_Observations_nao"]
            == coverage["Days_With_Observations_all"],
            "Week_Start",
        ]

        weekly_nao_df = weekly_nao_df[
            weekly_nao_df["Week_Start"].isin(complete_weeks)
        ].copy()

    # Final safeguard: never export missing NAO values in this file.
    if "NAO_index" in weekly_nao_df.columns:
        weekly_nao_df = weekly_nao_df.dropna(
            subset=["NAO_index"]
        ).copy()

    weekly_nao_df.to_csv(
        output_with_nao,
        index=False
    )

    print("\n---------- WEEKLY AGGREGATION COMPLETE ----------")
    print(f"Weekly dataset without NAO: {output_without_nao}")
    print(f"Weekly dataset with NAO:    {output_with_nao}")
    print(f"Weeks without NAO: {len(weekly_df)}")
    print(f"Weeks with NAO:    {len(weekly_nao_df)}")

    if not weekly_nao_df.empty and weekly_nao_df["NAO_index"].isna().any():
        raise RuntimeError(
            "The weekly dataset with NAO unexpectedly contains missing values."
        )

    return weekly_df, weekly_nao_df


# ---------------------------------------------------------------------
# 5. Run as a standalone script
# ---------------------------------------------------------------------

if __name__ == "__main__":
    generate_weekly_datasets()
