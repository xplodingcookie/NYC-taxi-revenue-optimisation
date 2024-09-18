from urllib.request import urlretrieve
import os
import sys
sys.path.append("../")
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, stddev, mean, col, to_date, concat
import numpy as np

def get_weather_data():
    """Get the weather data records for 2022-10-01 to 2023-03-31 for new york central park and
    output them to the data/landing folder"""

    output_relative_dir = "../data/"

    # Check if it exists as it makedir will raise an error if it does exist
    if not os.path.exists(output_relative_dir):
        os.makedirs(output_relative_dir)
        
    # Create new path
    if not os.path.exists(output_relative_dir + "landing/noaa_data"):
        os.makedirs(output_relative_dir + "landing/noaa_data")
    if not os.path.exists(output_relative_dir + "raw/noaa_data"):
        os.makedirs(output_relative_dir + "raw/noaa_data")

    # The years to obtain data from
    years = ["2022", "2023"]

    # This is the URL template for weather and ID for NY CITY CENTRAL PARK data as of 08/2024
    STATION_ID = "GHCNh_USW00094728"
    url_template = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/"

    # There is no schema conversion needed landing and raw will be identical
    for write_location in ['landing/', 'raw/']:
        for year in years:
            print(f"Begin {year} for weather")
            # Make URL for the month
            url = f"{url_template}{year}/psv/{STATION_ID}_{year}.psv"
            # Generate output location and filename
            output_dir = f"{output_relative_dir}{write_location}noaa_data/{year}.psv"
            print(url)
            print(output_dir)
            try:
                # Download psv file
                urlretrieve(url, output_dir)
                print(f"Completed {year} for weather")
            except Exception as e:
                print(f"Failed to download data for {year}: {e}")

def preprocess_future_weather_data():
    """Pre-process the weather data the same way that is described in preprocess_weather_ data,
    this will be used as a funciton to clean the future data for predicting"""
    # create a spark session (which will run spark jobs)
    spark = (
        SparkSession.builder.appName("Preprocess Weather Data")
        .config("spark.sql.repl.eagerEval.enabled", True) 
        .config("spark.sql.parquet.cacheMetadata", "true")
        .config("spark.sql.session.timeZone", "Etc/UTC")
        .getOrCreate()
    )

    year = "2023"

    weather_df = spark.read.format("csv") \
            .option("delimiter", "|") \
            .option("header", "true") \
            .load(f"../data/raw/noaa_data/{year}.psv")


    attributes = [
        "Year",
        "Month",
        "Day",
        "temperature",
        "dew_point_temperature",
        "station_level_pressure",
        "sea_level_pressure",
        "wind_speed",
        "precipitation",
        "relative_humidity",
        "wet_bulb_temperature"
    ]

    weather_df = weather_df.select(attributes)

    weather_df = weather_df.withColumn(
        "date",
        to_date(
            concat(
                col("Year"), 
                lit("-"), 
                col("Month"), 
                lit("-"), 
                col("Day")
            ), 
            "yyyy-MM-dd"
        )
    )

    # define the start and end dates
    start_date = "2023-04-01"
    end_date = "2023-06-30"

    # filter df to only include the specified date range
    weather_df = weather_df.filter(
        (col("date") >= start_date) & (col("date") <= end_date)
    )

    for column in attributes:
        stats = weather_df.agg(
            mean(column).alias("mean"),
            stddev(column).alias("stddev")
        ).collect()[0]
        
        column_mean = stats["mean"]
        column_stddev = stats["stddev"]

        bound_sd = np.sqrt(2*np.log(weather_df.count()))

        weather_df = weather_df.filter(
            (col(column) >= column_mean - bound_sd * column_stddev) &
            (col(column) <= column_mean + bound_sd * column_stddev)
        )

    daily_weather_df = weather_df.groupBy("date").agg(
        mean("temperature").alias("mean_temperature"),
        mean("dew_point_temperature").alias("mean_dew_point_temperature"),
        mean("station_level_pressure").alias("mean_station_level_pressure"),
        mean("sea_level_pressure").alias("mean_sea_level_pressure"),
        mean("wind_speed").alias("mean_wind_speed"),
        mean("precipitation").alias("mean_precipitation"),
        mean("relative_humidity").alias("mean_relative_humidity"),
        mean("wet_bulb_temperature").alias("mean_wet_bulb_temperature")
    )

    daily_weather_df = daily_weather_df.orderBy("date")

    daily_weather_df.write.mode('overwrite').parquet('../data/curated/future_weather_data.parquet')