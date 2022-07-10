"""Feature engineering the dataset."""
import argparse
import logging
import os
import pathlib
import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

    
if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/books.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn, error_bad_lines = False)
    os.unlink(fn)

    logger.info("Feature engineering average_rating column")
    df.loc[ (df['average_rating'] >= 0) & (df['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
    df.loc[ (df['average_rating'] > 1) & (df['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
    df.loc[ (df['average_rating'] > 2) & (df['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
    df.loc[ (df['average_rating'] > 3) & (df['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
    df.loc[ (df['average_rating'] > 4) & (df['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

    rating_df = pd.get_dummies(df['rating_between'])
    language_df = pd.get_dummies(df['language_code'])

    logger.info("Creating training features")
    features = pd.concat([rating_df, 
                    language_df, 
                    df['average_rating'], 
                    df['ratings_count']], axis=1)

    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)

    logger.info("Writing out datasets to %s.", base_dir)
    np.save(f"{base_dir}/train/train.npy", features)
