"""Module of functions for exporting data to the designated Aurora-PostGres DB tables"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from psycopg2 import sql

import config_class.mde.premarket as mde_premarket
from config_class.mde.logger import define_logger as logger

home_dir = os.getcwd()
config = mde_premarket.Config(os.path.join(home_dir, "config.yml"))


def parse_to_datetime(val):
    """
    Parses the date time value 
    """
    if isinstance(val, datetime):
        return val
    elif isinstance(val, str):
        try:
            return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                return datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                return None  # Or raise
    return None
    
def insert_reason_for_submission_data(rs_df):
    """
    Inserts (row-by-row) the processed files for reason for submission metadataframe into its postgres table
    params: rs_df - A dataframe of OIMT documents processed by reason-for-submission extraction pipeline
    """
    log = logger()
    # db configuration object
    from data.db_config import get_db_connection

    # RDS table and schema
    reason_for_submission_schema = config["DB"]
    table = config["TABLE"]

    conn = get_db_connection()
    cur = conn.cursor()

    # Ensure the schema is included in the search path
    cur.execute(f"SET search_path TO {reason_for_submission_schema};")

    for idx, row in rs_df.iterrows():
        try:
            # Convert each field as needed, with fallback to None where applicable
            unique_id = str(row.get("unique_id")) if pd.notnull(row.get("unique_id")) else None
            doc_id = str(row.get("doc_id")) if pd.notnull(row.get("doc_id")) else None
            submission_num = str(row.get("submission_num")) if pd.notnull(row.get("submission_num")) else None
            num_of_pages = int(row.get("num_of_pages")) if pd.notnull(row.get("num_of_pages")) else None
            submissiontype = str(row.get("submissiontype")) if pd.notnull(row.get("submissiontype")) else None
            pdf_s3_url = str(row.get("pdf_s3_url")) if pd.notnull(row.get("pdf_s3_url")) else None
            object_name = str(row.get("object_name")) if pd.notnull(row.get("object_name")) else None
            reason = str(row.get("reason")) if pd.notnull(row.get("reason")) else None
            found = str(row.get("found")) if pd.notnull(row.get("found")) else None
            page = int(row["page"]) if pd.notnull(row.get("page")) else None
            relevancy_score = float(row["relevancy_score"]) if pd.notnull(row.get("relevancy_score")) else None
            ocr = str(row.get("ocr")) if pd.notnull(row.get("ocr")) else None
            coordinates = str(row.get("coordinates")) if pd.notnull(row.get("coordinates")) else None
            date_updated = parse_to_datetime(row.get("date_updated")) if pd.notnull(row.get("date_updated")) else None

            # Use psycopg2.sql to safely format the table name
            insert_query = sql.SQL(
                """
                INSERT INTO {table} (
                    unique_id, doc_id, submission_num, num_of_pages, submissiontype, pdf_s3_url, object_name, reason, found, page, relevancy_score, ocr,
                    coordinates, date_updated
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            ).format(table=sql.Identifier(table))

            # Execute with parameterized values
            cur.execute(
                insert_query,
                (
                    unique_id,
                    doc_id,
                    submission_num,
                    num_of_pages,
                    submissiontype,
                    pdf_s3_url,
                    object_name,
                    reason,
                    found,
                    page,
                    relevancy_score,
                    ocr,
                    coordinates,
                    date_updated,
                ),
            )

            # Commit the transaction
            conn.commit()
            log.info(f"Succesfully inserted into {reason_for_submission_schema}.{table}")
        except Exception as e:
            # Roll back the transaction on error
            conn.rollback()
            log.error(f"Error inserting data: {e}")


def load_data_with_reason(
    unique_id,
    docid,
    source_submission_num,
    num_of_pages,
    r_creation_date,
    r_modify_date,
    submissiontype,
    pdf_s3_url,
    object_name,
    reason,
    found,
    page,
    relevancy_score,
    ocr,
    coordinates,
):
    """
    Loads the reason-for-submission extraction pipeline outputs into the designated Aurora postgres table.
    Integrated function of Dataframe conversion and table insertion
    params -  Metadata of processed/extraction data
    """
    load_df_dict = {
        "unique_id": unique_id,
        "doc_id": docid,
        "submission_num": source_submission_num,
        "num_of_pages": num_of_pages,
        "r_creation_date": r_creation_date,
        "r_modify_date": r_modify_date,
        "submissiontype": submissiontype,
        "pdf_s3_url": pdf_s3_url,
        "object_name": object_name,
        "reason": reason,
        "found": found,
        "page": page,
        "relevancy_score": relevancy_score,
        "ocr": ocr,
        "coordinates": coordinates,
        "date_updated": datetime.now(),
    }
    # Create a pandas df from the dictionary data structure
    load_df = pd.DataFrame(load_df_dict, index=[0])

    # Insert the dataframe into gold_mde_output.ri2_reason_for_submission_llm
    insert_reason_for_submission_data(load_df)

    # Just print to console for dev/testing
    #print(load_df)