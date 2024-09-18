from pyspark.sql.functions import col
from pyspark.sql.types import LongType, IntegerType, DoubleType, StringType

def taxi_type_correction(df):
    """ Corrects type inconsistencies in the dataframe """
    df.withColumn("VendorID", col("VendorID").cast(IntegerType())) \
        .withColumn("passenger_count", col("passenger_count").cast(DoubleType())) \
        .withColumn("trip_distance", col("trip_distance").cast(DoubleType())) \
        .withColumn("RatecodeID", col("RatecodeID").cast(DoubleType())) \
        .withColumn("store_and_fwd_flag", col("store_and_fwd_flag").cast(StringType())) \
        .withColumn("PULocationID", col("PULocationID").cast(LongType())) \
        .withColumn("DOLocationID", col("DOLocationID").cast(LongType())) \
        .withColumn("payment_type", col("payment_type").cast(DoubleType())) \
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType())) \
        .withColumn("extra", col("extra").cast(DoubleType())) \
        .withColumn("mta_tax", col("mta_tax").cast(DoubleType())) \
        .withColumn("tip_amount", col("tip_amount").cast(DoubleType())) \
        .withColumn("tolls_amount", col("tolls_amount").cast(DoubleType())) \
        .withColumn("improvement_surcharge", col("improvement_surcharge").cast(DoubleType())) \
        .withColumn("total_amount", col("total_amount").cast(DoubleType())) \
        .withColumn("congestion_surcharge", col("congestion_surcharge").cast(DoubleType()))
    return df