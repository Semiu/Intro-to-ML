"""Module for functions used in importing the OIMT metadata for the reason for submission task's documents"""

import pandas as pd
import os
import re
import boto3
from botocore.exceptions import ClientError
from datetime import datetime

from data_generation.preprocessing import preprocess_text
import config_class.mde.premarket as mde_premarket
from config_class.mde.assume_role import assume_role_with_arn
from config_class.mde.logger import define_logger as logger

# Read the configuration file
ROLE_ARN = "arn:aws-us-gov:iam::098592471684:role/CEDH-Cross-Account-Role-for-MDEAI"  # cross-account role allowing access to CEDh OIMT bucket
home_dir = os.getcwd()
config = mde_premarket.Config(os.path.join(home_dir, "config.yml"))
cutoff = pd.to_datetime(
    "2024-10-10"
)  # cutoff date for files in CEDh OIMT bucket -- before 20241010 files are stored in root dir, after 20241010 stored in subfolders

assumed_session = assume_role_with_arn(ROLE_ARN)


def get_s3_path(bucket_name, data):
    """
    Args:
     bucket_name (str): OIMT S3 bucket name
     data (dict): Data dictionary for the the document metadata
    Returns:
     s3_path (str): S3 path of the document file
    """
    doc_id = data["r_object_id"]
    try:
        k = data["folder_id"]
    except KeyError as e:
        k = data["source_submission_num"]

    download_date = data["doc_download_date"]
    # If it's a string, convert it to a datetime object
    if isinstance(download_date, str):
        try:
            download_date = datetime.strptime(download_date, "%Y-%m-%d")
        except ValueError:
            # Try a more detailed format if needed
            download_date = datetime.strptime(download_date, "%Y-%m-%d %H:%M:%S")

    date_formatted = download_date.strftime("%Y%m%d")

    if download_date < cutoff:  # if before cutoff, look in root directory
        key = f"lz_i2k/{doc_id}"
    else:  # if on or after cutoff, look in subfolders
        key = f"lz_i2k/{date_formatted}/{k}/{doc_id}"

    s3_path = f"s3://{bucket_name}/{key}"
    return s3_path


def download_file_from_s3(bucket_name, metadata, local_path):
    """
    Downloads a file from an S3 bucket to the local filesystem.
    Args:
    - bucket_name (str): Name of the S3 bucket.
    - metadata (dict): Data dictionary for the the document metadata
    - local_path (str): Local path where the downloaded file will be saved.
    """
    log = logger()

    s3 = assumed_session.client("s3")

    doc_id = metadata["r_object_id"]
    try:
        k = metadata["folder_id"]
    except KeyError as e:
        k = metadata["source_submission_num"]

    download_date = metadata["doc_download_date"]
    # If it's a string, convert it to a datetime object
    if isinstance(download_date, str):
        try:
            download_date = datetime.strptime(download_date, "%Y-%m-%d")
        except ValueError:
            # Try a more detailed format if needed
            download_date = datetime.strptime(download_date, "%Y-%m-%d %H:%M:%S")

    date_formatted = download_date.strftime("%Y%m%d")

    if download_date < cutoff:  # if before cutoff, look in root directory
        key = f"lz_i2k/{doc_id}"
    else:  # if on or after cutoff, look in subfolders
        key = f"lz_i2k/{date_formatted}/{k}/{doc_id}"

    try:
        s3.download_file(bucket_name, key, local_path)
    except ClientError as e:
        log.error(f"Error downloading file from S3: {e}")


def inscope_documents(sentence):
    """
    Checks for documents in scope (among the documents returned from the OIMT metadata table)
    Args:
        sentence (str) - the object name in the OIMT metadata field
    Returns:
        True or False (bool) - the check for the existence of the keyword
    """
    keyword_sets = config["KEYWORD_SETS"]
    for keywords in keyword_sets:
        # Check if all keywords in the current list are present in the sentence
        if all(
            keyword.lower() in preprocess_text(sentence.lower()).replace(" ", "")
            for keyword in keywords
        ):
            return True

    return False


def get_sort_key(row):
    """
    Gets sorting key based on substring match

    params: data dictionary
    returns: order length
    """
    # Sorting guide provided in the configuration file
    sorting_order = config["SORTING_ORDER"]

    clean_column4 = "".join(c.lower() for c in row["object_name"] if c.isalnum())
    for substring, order in sorting_order.items():
        clean_substring = "".join(c.lower() for c in substring if c.isalnum())
        if clean_substring in clean_column4:
            return order
    return len(sorting_order)


def get_rs_document_ids(df):
    """
    Gets the r_object_id list from the documents to be processed for reason-for-submission
    Args:
        df - A dataframe of OIMT metadata of documents
    Returns:
        document_ids - A list of r_object_id of the documents
    """
    document_ids = list(df["r_object_id"])
    return document_ids


def get_extracted_text_df(ids):
    """
    Gets the extracted text df
    Args:
        ids: List of ids - r_object_id
    Returns:
        extracted_data_df: dataframe of the extracted text
    """
    ids_tuple = tuple(ids)

    # Query for fetching data
    from data.queries import get_extracted_text_query

    # Db configuration object
    from data.db_config import get_db_connection

    # Get the query string - unprocessed data for ri2_reason_for_submission_llm
    text_query = get_extracted_text_query(ids_tuple)

    # Get the database connection object
    conn_pg = get_db_connection()

    extracted_data_df = pd.read_sql_query(text_query, conn_pg)

    return extracted_data_df


def get_rs_metadata_df():
    """
    Gets OIMT metadata dataframe for the reason for submission LLM-based question answering task

    Returns
        unprocessed_oimt_data_df - a dataframe of unprocessed data based on record in gold_mde_output.ri2_reason_for_submission_llm
    """
    # Query for fetching data
    from data.queries import get_reason_for_submission_files

    # Db configuration object
    from data.db_config import get_db_connection

    # Get the query string - unprocessed data for ri2_reason_for_submission_llm
    rs_query = get_reason_for_submission_files()

    # Get the database connection object
    conn_pg = get_db_connection()

    return pd.read_sql_query(rs_query, conn_pg)


def get_inscope_df(oimt_df):
    """
    Gets inscope documents dataframe by filtering our documents not in scope
    Args:
        oimt_df (df) - Dataframe of OIMT metadata
    Returns
        Dataframe exclusively for documents that are in scope of the reason for submission LLM extraction task
    """
    # Iterate through the DataFrame, filter, and get the inscope documents
    group_df = oimt_df[oimt_df["object_name"].apply(inscope_documents)]
    if group_df.empty:
        regex_result_df = pd.DataFrame()
        return regex_result_df

    # Sort Dataframe by creation date ensuring most recent documents are at top
    # Convert 'r_creation_date' column from string to datetime
    group_df.loc[:, "r_creation_date"] = pd.to_datetime(group_df["r_creation_date"])

    # Sort the DataFrame by 'r_creation_date' column
    group_df = group_df.sort_values(by="r_creation_date", ascending=False)

    # Add a temporary column for sorting
    group_df = group_df.copy()  # Create a copy of the DataFrame

    group_df["sort_key"] = group_df.apply(get_sort_key, axis=1)  # Modify the copy

    group_df.loc[:, "sort_key"] = group_df.sort_values(by="sort_key")["sort_key"]

    return group_df
