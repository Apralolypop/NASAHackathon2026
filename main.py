
import streamlit as st
from datetime import date

from pipeline import generate_dataset


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NASA Environmental Dataset Generator",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌱 Environmental Dataset Generator")

st.write(
    """
    Generate machine-learning-ready environmental datasets
    using historical NASA POWER data.

    Enter a location, select a date range, and generate
    a CSV dataset for further analysis and model training.
    """
)

st.divider()


# --------------------------------------------------
# LOCATION INPUTS
# --------------------------------------------------

st.subheader("1. Location")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=48.424274,
        format="%.6f"
    )

with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-104.486416,
        format="%.6f"
    )


# --------------------------------------------------
# DATE INPUTS
# --------------------------------------------------

st.divider()
st.subheader("2. Date range")

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start date",
        value=date(2025, 1, 1)
    )

with col2:
    end_date = st.date_input(
        "End date",
        value=date(2025, 12, 31)
    )


# --------------------------------------------------
# SAMPLING INTERVAL
# --------------------------------------------------

st.divider()
st.subheader("3. Sampling interval")

interval = st.selectbox(
    "Retrieve one observation every:",
    options=[1, 2, 3, 4, 6, 12, 24],
    index=0,
    format_func=lambda x: f"{x} hour(s)"
)

st.caption(
    "For example, selecting 1 hour keeps every hourly "
    "observation, while 6 hours keeps one observation "
    "every six hours."
)

st.subheader("Additional climate data")

include_prism = st.checkbox(
    "Include PRISM data",
    value=False
)

prism_variables = []

if include_prism:

    prism_variables = st.multiselect(
        "PRISM variables",
        options=[
            "tmin",
            "tmax",
            "tmean",
            "ppt"
        ],
        default=[
            "tmin",
            "tmax",
            "ppt"
        ]
    )

    st.info(
        "PRISM provides daily data and currently covers "
        "the contiguous United States through this API."
    )


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.divider()
st.subheader("4. Dataset information")

st.write(
    """
    The dataset includes environmental parameters such as:
    - Air temperature
    - Dew point temperature
    - Relative humidity
    - Specific humidity
    - Atmospheric pressure
    - Wind speed and direction
    - Precipitation
    - Solar radiation
    """
)

st.info(
    "The generated CSV will be saved in the project's "
    "'data' folder."
)


# --------------------------------------------------
# GENERATE DATASET
# --------------------------------------------------

st.divider()

if st.button(
    "Generate CSV dataset",
    type="primary",
    use_container_width=True
):

    if start_date > end_date:
        st.error("The start date must be before the end date.")

    else:
        try:

            with st.spinner(
                "Retrieving and processing NASA POWER data..."
            ):

                df = generate_dataset(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    include_prism=include_prism,
                    prism_variables=prism_variables
                )

            st.success("Dataset successfully generated!")

            # Dataset statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Observations",
                    len(df)
                )

            with col2:
                st.metric(
                    "Variables",
                    len(df.columns) - 3
                )

            with col3:
                st.metric(
                    "Sampling interval",
                    f"{interval} hour(s)"
                )

            # Preview
            st.subheader("Dataset preview")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            # Missing data
            st.subheader("Missing data")

            missing_data = df.isna().sum()

            st.dataframe(
                missing_data.rename("Missing values"),
                use_container_width=True
            )

            # CSV download
            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="nasa_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success(
                "A copy of the dataset has also been saved "
                "locally in the data folder."
            )

        except Exception as error:

            st.error(
                f"Dataset generation failed: {error}"
            )
