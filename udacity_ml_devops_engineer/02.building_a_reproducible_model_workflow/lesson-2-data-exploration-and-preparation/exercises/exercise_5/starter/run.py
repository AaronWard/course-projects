#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    ## YOUR CODE HERE #"exercise_4/genres_mod.parquet:latest"
    artifact = run.use_artifact(args.input_artifact)
    df = pd.read_parquet(artifact.file(), engine="pyarrow")

    logger.info("dropping duplicate rows...")
    df = df.drop_duplicates().reset_index(drop=True)

    """
    NOTE: again, in a real setting, you will have to make sure that your feature store
    provides this text_feature at inference time, OR, you will have to move
    the computation of this feature to the inference pipeline.
    """
    logger.info("performing feature engineering...")
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    file_name = 'preprocessed_data.csv'
    df.to_csv(file_name)


    logger.info("Creating artifact for dataframe..")
    artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description
        )
    artifact.add_file(file_name)

    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
