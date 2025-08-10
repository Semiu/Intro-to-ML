"""Main module"""

import os
import pandas as pd
import numpy as np

from data_generation.import_data import (
    get_inscope_df,
    get_rs_document_ids,
    get_rs_metadata_df,
    get_extracted_text_df,
)

from response_production.get_reason import return_reason_for_submission
from data_generation.export_data import load_data_with_reason
from config_class.mde.logger import define_logger as logger

log = logger()


def main(inscope_df):
    """
    The main function
    params: inscope_df - Dataframe of the documents in scope as defined by the get_inscope_df function
    Returns: None
    """
    ## Iterate through the folder id groups
    for group_name, group_df in inscope_df:

        if group_df.empty:
            log.info("Dataframe is empty. No data to process")
            continue

        # Calls the RAG-LLM module to retrieve the reason for submission
        reason = return_reason_for_submission(group_df)

        if reason is None:
            # Taking one of the folder_id document as a representative of the submission
            none_reason_df = group_df.head(1)

            submission_id = none_reason_df["folder_id"].values[0]
            object_id = none_reason_df["r_object_id"].values[0]
            submission_type = none_reason_df["folder_type"].values[0]
            unique_id = f"""{submission_id.lower()}-{object_id.replace('.pdf', '')}"""

            load_data_with_reason(
                unique_id,
                object_id,
                submission_id,
                np.nan,
                np.nan,
                np.nan,
                submission_type,
                np.nan,
                np.nan,
                "Reason not found",
                "N",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            log.info("Nonetype reason is returned")
        else:
            # Load the reason to the designated PostgreSQL DB table
            load_data_with_reason(
                reason["unique_id"],
                reason["r_object_id"],
                reason["source_submission_num"],
                reason["number_of_pages"],
                reason["r_creation_date"],
                reason["r_modify_date"],
                reason["submissiontype"],
                reason["s3_pdf_path"],
                reason["object_name"],
                reason["reason"],
                "Y",
                reason["page"],
                reason["relevancy_score"],
                reason["ocr"],
                reason["coordinates"],
            )


if __name__ == "__main__":

    try:
        log.info("Process starts")
        # Get data
        oimt_df = get_rs_metadata_df()
        # Get the reason for submission metadata df
        log.info(f"OIMT data df out - length {len(oimt_df)}")

        # Get the list of object_ids to be processed
        object_ids = get_rs_document_ids(oimt_df)
        log.info(f"unprocessed {len(object_ids)} object IDs grabbed")

        # Get the corresponding extracted text from the to-be-processed object_ids
        document_extracted_text_df = get_extracted_text_df(object_ids)

        if not document_extracted_text_df.empty:

            log.info(
                f"Text extract DF grabbed of total {len(document_extracted_text_df)}"
            )

            # Join oimt_df and document_extracted_text_df to have full metadata
            full_metadata_df = pd.merge(
                document_extracted_text_df,
                oimt_df,
                how="left",
                left_on="document_id",
                right_on="r_object_id",
            )
            log.info(f"Dfs length {len(full_metadata_df)} merged")

            # Get the reason for submission metadata df for documents in scope
            inscope_df = get_inscope_df(full_metadata_df)

            if not inscope_df.empty:
                inscope_df_grouped = inscope_df.groupby("folder_id")
                main(inscope_df_grouped)
                log.info("Process succesfully completed!")
            else:
                log.info("The inscope df is empty")
        else:
            log.info("No available text extract to process")
    except Exception as e:
        log.error(f"Process failed to complete because of {e}")
