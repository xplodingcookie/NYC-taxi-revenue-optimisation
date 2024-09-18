from urllib.request import urlretrieve
import os

def get_traffic_data():
    """Get the traffic data records for 2022-10-01 to 2023-03-31 for new york central park and
    output them to the data/landing and data/raw folder"""

    for write_location in ['landing', 'raw']:
        output_relative_dir = f"../data/{write_location}/"

        # Check if it exists as it makedir will raise an error if it does exist
        if not os.path.exists(output_relative_dir):
            os.makedirs(output_relative_dir)
            
        # Create new path
        if not os.path.exists(output_relative_dir + "traffic_data"):
            os.makedirs(output_relative_dir + "traffic_data")

        # Url as of 11/08
        url = "https://data.cityofnewyork.us/resource/7ym2-wayt.csv?$limit=1673725"

            # Generate output location and filename
        output_dir = f"{output_relative_dir}traffic_data/traffic_data.csv"
        print(url)
        print(output_dir)
        try:
            # Download csv file
            urlretrieve(url, output_dir)
            print(f"Completed traffic data download")
        except Exception as e:
            print(f"Failed to download traffic data: {e}")