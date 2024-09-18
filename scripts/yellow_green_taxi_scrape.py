from urllib.request import urlretrieve
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, stddev, mean, col, unix_timestamp, abs, round, to_date, count, when, dayofweek, sum
import numpy as np
from scripts.utility import taxi_type_correction

def get_taxi_data():
    """Get the yellow and green taxi data records for 2022-10-01 to 2023-03-31 and output them to
    the data/landing folder"""

    output_relative_dir = "../data/landing/"

    # Check if it exists as it makedir will raise an error if it does exist
    if not os.path.exists(output_relative_dir):
        os.makedirs(output_relative_dir)
        
    # Create new paths
    if not os.path.exists(output_relative_dir + "tlc_data"):
        os.makedirs(output_relative_dir + "tlc_data")
    if not os.path.exists(output_relative_dir + "/raw/tlc_data"):
        os.makedirs(output_relative_dir + "/raw/tlc_data")

    # The years and months to obtain data from
    time_period = {
        "2022": [10, 11, 12],
        "2023": [1, 2, 3]
    }

    # These are the URL template for yellow and green taxi data as of 08/2024
    URL_TEMPLATES = {
        "yellow_taxi": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_",
        "green_taxi": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_"
    }

    # Generate output directory for each type of taxi respectively
    for taxi_type in URL_TEMPLATES.keys():
        if not os.path.exists(output_relative_dir + "tlc_data/" + taxi_type):
            os.makedirs(output_relative_dir + "tlc_data/" + taxi_type)
        if not os.path.exists("../data/raw/tlc_data/" + taxi_type):
            os.makedirs("../data/raw/tlc_data/" + taxi_type)

    for taxi_type, url_template in URL_TEMPLATES.items():
        for year, months in time_period.items():
            for month in months:
                month = str(month).zfill(2)
                print(f"Begin {month}/{year} for {taxi_type}")

                # Make URL for the month
                url = f"{url_template}{year}-{month}.parquet"
                # Generate output location and filename
                output_dir = f"{output_relative_dir}tlc_data/{taxi_type}/{year}-{month}.parquet"
                # Download parquet file
                urlretrieve(url, output_dir)

                print(f"Completed {month}/{year} for {taxi_type}")

def get_taxi_data_future():
    """Get the yellow and green taxi data records for 2023-04-01 to 2023-06-30 and output them to
    the data/raw folder"""

    output_relative_dir = "../data/raw/"

    # Check if it exists as it makedir will raise an error if it does exist
    if not os.path.exists(output_relative_dir):
        os.makedirs(output_relative_dir)
        
    # Create new path
    if not os.path.exists(output_relative_dir + "tlc_data"):
        os.makedirs(output_relative_dir + "tlc_data")

    # The years and months to obtain data from
    time_period = {
        "2023": [4, 5, 6]
    }

    # These are the URL template for yellow and green taxi data as of 08/2024
    URL_TEMPLATES = {
        "yellow_taxi": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_",
        "green_taxi": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_"
    }

    # Generate output directory for each type of taxi respectively
    for taxi_type in URL_TEMPLATES.keys():
        if not os.path.exists(output_relative_dir + "tlc_data/" + taxi_type):
            os.makedirs(output_relative_dir + "tlc_data/" + taxi_type)

    for taxi_type, url_template in URL_TEMPLATES.items():
        for year, months in time_period.items():
            for month in months:
                month = str(month).zfill(2)
                print(f"Begin {month}/{year} for {taxi_type}")

                # Make URL for the month
                url = f"{url_template}{year}-{month}.parquet"
                # Generate output location and filename
                output_dir = f"{output_relative_dir}tlc_data/{taxi_type}/{year}-{month}.parquet"
                # Download parquet file
                urlretrieve(url, output_dir)

                print(f"Completed {month}/{year} for {taxi_type}")

def preprocess_future_taxi_data():
    """Pre-process the taxi data the same way that is described in preprocess_taxi_ data,
    this will be used as a funciton to clean the future data for predicting"""
    spark = (
    SparkSession.builder.appName("Preprocess Taxi Data")
    .config("spark.sql.repl.eagerEval.enabled", True) 
    .config("spark.sql.parquet.cacheMetadata", "true")
    .config("spark.sql.session.timeZone", "Etc/UTC")
    .getOrCreate()
    )

    time_period = {
        "2023": [4, 5, 6]
    }

    # read in the green taxi parquet files individually and correctly type cast them
    green_sdf_list = []
    for year, months in time_period.items():
        for month in months:
            month_str = str(month).zfill(2)
            file_path = f"../data/raw/tlc_data/green_taxi/{year}-{month_str}.parquet"
            df = spark.read.parquet(file_path)
            # type cast to correct type 
            df_casted = taxi_type_correction(df)
            green_sdf_list.append(df_casted)

    # read in the yellow taxi parquet files individually and correctly type cast them and correct
    # column name inconsistencies
    yellow_sdf_list = []
    for year, months in time_period.items():
        for month in months:
            month_str = str(month).zfill(2)
            file_path = f"../data/raw/tlc_data/yellow_taxi/{year}-{month_str}.parquet"
            df = spark.read.parquet(file_path)
            df_casted = taxi_type_correction(df)
            df_casted.withColumnRenamed("Airport_fee", "airport_fee")
            yellow_sdf_list.append(df_casted)

    # concatenate the yellow and green taxi dataframes respectively
    yellow_sdf = yellow_sdf_list[0]
    green_sdf = green_sdf_list[0]
    for df in yellow_sdf_list[1:]:
        yellow_sdf = yellow_sdf.union(df)
    for df in green_sdf_list[1:]:
        green_sdf = green_sdf.union(df)

    column_name = {"VendorID": "vendor_id",
               "RatecodeID": "ratecode_id",
               "PULocationID": "pu_location_id",
               "DOLocationID": "do_location_id"}

    for key, value in column_name.items():
        green_sdf = green_sdf.withColumnRenamed(key, value)
        yellow_sdf = yellow_sdf.withColumnRenamed(key, value)

    # Drop ehail fee and add a column called airport fee with values initialised to 0
    green_sdf = green_sdf.drop("ehail_fee", "trip_type", "store_and_fwd_flag")
    yellow_sdf = yellow_sdf.drop("store_and_fwd_flag")
    green_sdf = green_sdf.withColumn("airport_fee", lit(0))

    # Rename the datetime columns to match
    green_sdf = (green_sdf.withColumnRenamed("lpep_pickup_datetime", "pep_pickup_datetime")
                        .withColumnRenamed("lpep_dropoff_datetime", "pep_dropoff_datetime"))

    yellow_sdf = (yellow_sdf.withColumnRenamed("tpep_pickup_datetime", "pep_pickup_datetime")
                            .withColumnRenamed("tpep_dropoff_datetime", "pep_dropoff_datetime"))
    
    taxi_df = yellow_sdf.union(green_sdf)

    numeric_columns = [
        "vendor_id",
        "passenger_count",
        "trip_distance",
        "ratecode_id",
        "pu_location_id",
        "do_location_id",
        "payment_type",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "airport_fee"
    ]

    # Define date range
    start_date = "2022-04-01"
    end_date = "2023-06-30"

    # Filter DataFrame
    taxi_df = taxi_df.filter(
        (col("pep_pickup_datetime") >= start_date) &
        (col("pep_dropoff_datetime") <= end_date)
    )

    # Fix semantic errors in the data
    taxi_df = taxi_df.where(
                    (col("passenger_count") > 0) &
                    (col("trip_distance") > 0) &
                    (col("ratecode_id") >= 1) &
                    (col("ratecode_id") <= 6) &
                    (col("payment_type") >= 1) &
                    (col("payment_type") <= 6) &
                    (col("fare_amount") >= 2.50) &
                    (col("extra") >= 0) &
                    (col("mta_tax") >= 0) &
                    (col("tip_amount") >= 0) &
                    (col("tolls_amount") >= 0) &
                    (col("improvement_surcharge") >= 0) &
                    (col("total_amount") >= 0) &
                    (col("congestion_surcharge") >= 0) &
                    (col("airport_fee") >= 0) &
                    (col("airport_fee") <= 1.25)
                    )

    # Define trip time in hours and filter time that is too little or negative
    taxi_df = taxi_df.withColumn(
        "trip_time_hours",
        (unix_timestamp(col("pep_dropoff_datetime")) - unix_timestamp(col("pep_pickup_datetime"))) / (60*60)
    )

    taxi_df = taxi_df.where((col("trip_time_hours") > 0.02))

    # Find total_amount and calc_amount (calculated amount) difference to find errors in summation 
    taxi_df = taxi_df.withColumn(
                        "calc_total_amount",
                        col("fare_amount") + 
                        col("extra") + 
                        col("mta_tax") + 
                        col("tip_amount") + 
                        col("improvement_surcharge") + 
                        col("congestion_surcharge") +
                        col("tolls_amount") +
                        col("airport_fee"))

    taxi_df = taxi_df.withColumn(
        "total_diff",
        round(abs(col("total_amount") - col("calc_total_amount")), 4)
    )

    # As small errors are bound to happen and airport_fee is often seen to not be added we can leave them as long
    # as the total difference is <= 3
    taxi_df.filter(
        (col("total_diff") <= 3)
    )

    continuous_columns = [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra", "mta_tax",
        "tip_amount", 
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "airport_fee",
        "trip_time_hours"
    ]

    for column in continuous_columns:
        stats = taxi_df.agg(
            mean(column).alias("mean"),
            stddev(column).alias("stddev")
        ).collect()[0]
        
        column_mean = stats["mean"]
        column_stddev = stats["stddev"]

        if column_stddev is None:
            print(f"Warning: Standard deviation for column '{column}' is None.")
            continue  # Skip this column if stddev is None

        bound_sd = np.sqrt(2*np.log(taxi_df.count()))
        
        taxi_df = taxi_df.filter(
            (col(column) >= column_mean - bound_sd * column_stddev) &
            (col(column) <= column_mean + bound_sd * column_stddev)
        )

    taxi_df = taxi_df.withColumn("date", to_date(col("pep_pickup_datetime")))

    taxi_averages_df = (
        taxi_df.groupBy("pu_location_id", "date")
        .agg(
            count("*").alias("num_pickups"),
            mean((col("total_amount") + col("calc_total_amount")) / 2).alias("avg_total_amount"),
            mean("trip_time_hours").alias("avg_trip_time"),
            mean("fare_amount").alias("avg_fare_amount"),
            mean("passenger_count").alias("avg_passenger_count"),
            mean("trip_distance").alias("avg_trip_distance"),
            mean("extra").alias("avg_extra"),
            mean("mta_tax").alias("avg_mta_tax"),
            mean("tip_amount").alias("avg_tip_amount"),
            mean("tolls_amount").alias("avg_tolls_amount"),
            mean("improvement_surcharge").alias("avg_improvement_surcharge"),
            mean("congestion_surcharge").alias("avg_congestion_surcharge"),
            mean("airport_fee").alias("avg_airport_fee")
        )
    ).withColumn(
        "avg_amount_per_hour",
        col("avg_total_amount") / col("avg_trip_time")
    )

    # Second Stage: Compute total daily revenue by summing up avg_total_amount for each date and location
    daily_zone_revenue_df = (
        taxi_df.groupBy("pu_location_id", "date")
        .agg(
            sum((col("total_amount") + col("calc_total_amount")) / 2).alias("daily_zone_revenue")
        )
    )

    # Join the daily_zone_revenue_df back with the taxi_averages_df to include the daily_zone_revenue
    taxi_averages_df = taxi_averages_df.join(daily_zone_revenue_df, on=["pu_location_id", "date"], how="left")

    # Add days of the week (Mon = 1, Tues = 2, etc.)
    taxi_averages_df = taxi_averages_df.withColumn("day_of_week",
                    when(dayofweek(taxi_averages_df.date) == 1, 7)
                    .otherwise(dayofweek(taxi_averages_df.date) - 1))
    
    taxi_averages_df = taxi_averages_df.orderBy("date")
    
    taxi_df.write.mode('overwrite').parquet('../data/curated/cleaned_future_taxi_data.parquet')
    taxi_averages_df.write.mode('overwrite').parquet('../data/curated/future_taxi_averages_data.parquet')