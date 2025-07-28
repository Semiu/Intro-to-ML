import tempfile
import os
import json

import config_class.mde.premarket as mde_premarket
from config_class.mde.logger import define_logger as logger
from information_retrieval.retrieve_response import retriever_results
from data_generation.import_data import download_file_from_s3
from data_generation.extract_text import (
    process_pdf_page_with_pymupdf,
    process_pdf_page_with_pytesseract,
)

# Read the configuration file
home_dir = os.getcwd()
config = mde_premarket.Config(os.path.join(home_dir, "config.yml"))


def return_reason_for_submission(group_df):
    """
    Main function for reason-for-submission extraction
    params: group_df
    """
    log = logger()

    BUCKET_NAME = config["BUCKET_NAME"]

    try:
        metadata = retriever_results(group_df)
    
        if metadata is None:
            return None
        elif "reason not found" in metadata["reason"].strip().lower():
            return None
        with tempfile.TemporaryDirectory() as temp_dir:
                # Download the PDF file from S3 to a local temporary directory
                local_pdf_path = os.path.join(temp_dir, "temp_pdf_file.pdf")
                download_file_from_s3(BUCKET_NAME, metadata, local_pdf_path)
    
                if metadata["ocr"] == "No":
                    # Get reason coordinates using pymupdf and if any coordinate is None, use tesseract
                    reason_coordinates = process_pdf_page_with_pymupdf(metadata, local_pdf_path
                    )
                    # Get coordinates. check if any coordinate value from pymupdf is None, use tesseract
                else:
                    reason_coordinates = process_pdf_page_with_pytesseract(
                        local_pdf_path=local_pdf_path,
                        page_number=int(metadata["page"]),
                        target_text=metadata["document_section"],
                    )
    
                # Remove document section
                del metadata["document_section"]
    
                # Add coordinates to metadata
                metadata["coordinates"] = json.dumps(reason_coordinates)
        return metadata
    except Exception as e:
        log.error(
            f" An error {e} occured in getting the text coordinates for the return reason for submission"
        )
        return None
