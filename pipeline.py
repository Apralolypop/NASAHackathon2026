
from pathlib import Path

import pandas as pd

from APIs.NASApowerAPI import fetch_nasa_data
from APIs.PRISMAPI import fetch_prism_data
from APIs.ERA5API import fetch_era5_data
from APIs.NOAAAPI import fetch_nao_data

PARAMETERS = [
    "T2M",
    "T2MDEW",
    "T2MWET",
    "RH2M",
    "QV2M",
    "PS",
    "WS10M",
    "WD10M",
    "U10M",
    "V10M",
    "PRECTOT",
    "ALLSKY_SFC_SW_DWN"
]


PRISM_VARIABLES = [
    "tmin",
    "tmax",
    "tmean",
    "ppt"
]


VALID_INTERVALS = [1, 2, 3, 4, 6, 12, 24]


def normalize_timestamp(df):
    """
    Normalize timestamps to UTC, stored without timezone
    information so datasets can be merged consistently.
    """

    df = df.copy()

    df["Timestamp"] = (
        pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(None)
    )

    return df


def generate_dataset(
    latitude,
    longitude,
    start_date,
    end_date,
    interval=1 
):

    # 1. Validate inputs

    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")

    start_date = pd.Timestamp(start_date).date()
    end_date = pd.Timestamp(end_date).date()

    if start_date > end_date:
        raise ValueError("Invalid date range.")

    if interval not in VALID_INTERVALS:
        raise ValueError("Invalid sampling interval.")

    project_root = Path(__file__).resolve().parent

    data_folder = project_root / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    source_status = {}

    # 2. Retrieve NASA POWER data

    print("Retrieving NASA POWER data...")

    nasa_df = fetch_nasa_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        parameters=PARAMETERS,
        time_standard="UTC",
        community="AG"
    )

    if nasa_df.empty:
        raise ValueError(
            "NASA POWER returned no observations."
        )

    nasa_df = normalize_timestamp(nasa_df)

    nasa_df = nasa_df.dropna(
        subset=["Timestamp"]
    )

    if nasa_df["Timestamp"].duplicated().any():
        raise ValueError(
            "NASA POWER returned duplicate timestamps."
        )

    source_status["NASA POWER"] = (
        f"Retrieved {len(nasa_df)} hourly observations."
        )
    # 3. Retrieve monthly NOAA NAO data and combine with NASA POWER

    print("Retrieving monthly NOAA NAO data...")

    nao_df = fetch_nao_data(
        start_date=start_date,
        end_date=end_date
    )

    if nao_df.empty:

        nasa_df["NAO_index"] = float("nan")

        source_status["NOAA NAO"] = (
            "No NAO observations available for the requested dates."
        )

    else:

        # Create a common year-month identifier.

        nasa_df["_NAO_month"] = (
            nasa_df["Timestamp"].dt.to_period("M")
        )

        nao_df["_NAO_month"] = (
            nao_df["Timestamp"].dt.to_period("M")
        )

        # Merge monthly NAO values into hourly NASA POWER data.

        nasa_df = nasa_df.merge(
            nao_df[["_NAO_month", "NAO_index"]],
            on="_NAO_month",
            how="left",
            validate="many_to_one",
            sort=False
        )

        nasa_df.drop(
            columns=["_NAO_month"],
            inplace=True
        )

        matched_months = (
            nao_df["NAO_index"].notna().sum()
        )

        source_status["NOAA NAO"] = (
            f"Retrieved {len(nao_df)} monthly observations; "
            f"{matched_months} have valid index values."
        )


    # 4. Retrieve ERA5 hourly data

    print("Retrieving hourly ERA5 data...")

    era5_folder = data_folder / "era5_cache"

    era5_df = fetch_era5_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        output_folder=era5_folder
    )

    if era5_df.empty:
        raise ValueError(
            "ERA5 returned no observations."
        )

    era5_df = normalize_timestamp(era5_df)

    era5_df = era5_df.dropna(
        subset=["Timestamp"]
    )

    if era5_df["Timestamp"].duplicated().any():
        raise ValueError(
            "ERA5 returned duplicate timestamps."
        )

    source_status["ERA5"] = (
        f"Retrieved {len(era5_df)} hourly observations."
    )



    # 5. Combine NASA POWER and ERA5

    print("Combining NASA POWER and ERA5...")

    combined_df = pd.merge(
        nasa_df,
        era5_df,
        on="Timestamp",
        how="outer",
        validate="one_to_one",
        sort=True
    )

    combined_df["Latitude"] = latitude
    combined_df["Longitude"] = longitude

    # 6. Retrieve daily PRISM data

    print("Retrieving PRISM data...")

    prism_cache = data_folder / "prism_cache"

    prism_columns = [
        f"PRISM_{variable}"
        for variable in PRISM_VARIABLES
    ]

    try:

        prism_df = fetch_prism_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=PRISM_VARIABLES,
            cache_folder=prism_cache
        )

        if prism_df.empty:
            raise ValueError(
                "PRISM returned no observations."
            )

        prism_df = prism_df.copy()

        prism_df["Date"] = pd.to_datetime(
            prism_df["Date"],
            errors="coerce"
        ).dt.date

        prism_df = prism_df.dropna(
            subset=["Date"]
        )

        if prism_df["Date"].duplicated().any():
            raise ValueError(
                "PRISM returned duplicate daily observations."
            )

        # Make sure all expected PRISM columns exist.

        for column in prism_columns:
            if column not in prism_df.columns:
                prism_df[column] = float("nan")

        # Associate each hourly observation with its
        # corresponding calendar date.

        combined_df["Date"] = (
            combined_df["Timestamp"].dt.date
        )

        combined_df = combined_df.merge(
            prism_df[["Date"] + prism_columns],
            on="Date",
            how="left",
            validate="many_to_one"
        )

        source_status["PRISM"] = (
            f"Retrieved {len(prism_df)} daily observations."
        )

    except Exception as error:

        # Keep the dataset even if PRISM cannot be retrieved.
        # This can happen outside PRISM coverage or because
        # of a temporary API/network error.

        for column in prism_columns:
            combined_df[column] = float("nan")

        source_status["PRISM"] = (
            f"Could not retrieve PRISM data: {error}"
        )

    # 7. Remove temporary date column

    combined_df.drop(
        columns=["Date"],
        inplace=True,
        errors="ignore"
    )

    # 8. Apply sampling interval AFTER merging all sources

    combined_df = combined_df.dropna(
        subset=["Timestamp"]
    )

    if interval > 1:

        combined_df = combined_df[
            combined_df["Timestamp"].dt.hour % interval == 0
        ].copy()

    if combined_df.empty:
        raise ValueError(
            "No observations remain after filtering."
        )

    combined_df = (
        combined_df
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    # 9. Export combined CSV

    output_path = data_folder / "combined_dataset.csv"

    combined_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Combined dataset saved to: {output_path}"
    )

    print(
        f"Final observations: {len(combined_df)}"
    )

    # Make source information available to Streamlit.

    combined_df.attrs["source_status"] = source_status

    return combined_df
