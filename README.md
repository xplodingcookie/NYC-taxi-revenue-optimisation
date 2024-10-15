## README

**Author:** `Dong Li`

**Research Goal:** My research goal is to predict the daily average fare amount for taxi rides in New York City

**Timeline:** The timeline for the research area is 2022-10-01 to 2023-03-31.

Before running the code please,
1. Download and install all of the requirements in `requierments.txt`
2. Download the zone files from https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip and place into `data/raw/taxi_zone`.

To run the pipeline, please run in order:
1. `preprocess_taxi_data.ipynb`: This notebook downloads the raw data into the `data/landing` and `data/raw` directory and exports cleaned data to `data/curated`.
2. `preprocess_weather_data.ipynb`: This notebook downloads the raw data into the `data/landing` and `data/raw` directory and exports cleaned data to `data/curated`.
3. `preprocess_traffic_data.ipynb`: This notebook downloads the raw data into the `data/landing` and `data/raw` directory and exports cleaned data to `data/curated`.
3. `prelimary_plots.ipynb`: This notebook is used to model the curated data and do prelimary analysis.
4. `models_and_plots.ipynb`: This notebook is used to train and test the linear regression and neural network models. It also plots graphs and data relevant to these models.

**Extra notes**:
Please clone the repo on github into your local machine as the notebooks do not render properly on github. Thank you.
