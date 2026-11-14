
from pathlib import Path

import pandas as pd

from APIs.NASApowerAPI import fetch_nasa_data
from APIs.PRISMAPI import fetch_prism_data


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


def generate_dataset(
    latitude,
    longitude,
    start_date,
    end_date,
    interval=1,
    include_prism=False,
    prism_variables=None
):

    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")

    if start_date > end_date:
        raise ValueError("Invalid date range.")

    if interval not in [1, 2, 3, 4, 6, 12, 24]:
        raise ValueError("Invalid sampling interval.")

    # -----------------------------
    # 1. Retrieve NASA POWER data
    # -----------------------------

    nasa_df = fetch_nasa_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        parameters=PARAMETERS,
        time_standard="LST",
        community="AG"
    )

    if nasa_df.empty:
        raise ValueError("NASA returned no observations.")

    # -----------------------------
    # 2. Retrieve PRISM data
    # -----------------------------

    if include_prism:

        project_root = Path(__file__).resolve().parent

        cache_folder = (
            project_root / "data" / "prism_cache"
        )

        prism_df = fetch_prism_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=prism_variables,
            cache_folder=cache_folder
        )

        # Match each NASA timestamp to its calendar date.
        nasa_df["Date"] = (
            nasa_df["Timestamp"].dt.date
        )

        combined_df = nasa_df.merge(
            prism_df,
            on="Date",
            how="left"
        )

        combined_df.drop(
            columns=["Date"],
            inplace=True
        )

    else:
        combined_df = nasa_df

    # -----------------------------
    # 3. Apply sampling interval
    # -----------------------------

    if interval > 1:

        combined_df = combined_df[
            combined_df["Timestamp"].dt.hour % interval == 0
        ].copy()

    if combined_df.empty:
        raise ValueError("No observations remain.")

    # -----------------------------
    # 4. Export CSV
    # -----------------------------

    project_root = Path(__file__).resolve().parent

    data_folder = project_root / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    output_path = (
        data_folder / "combined_dataset.csv"
    )

    combined_df.to_csv(
        output_path,
        index=False
    )

    print(f"Dataset saved to: {output_path}")

    return combined_df
