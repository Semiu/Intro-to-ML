"""Module for the functions used in extracting text from the associated OIMT documents downloaded using the object_id field of the metadata"""

import tempfile
import os
import boto3
import fitz
from PIL import Image
from langchain.schema import Document
import pytesseract
import datetime
import nltk
from fuzzywuzzy import fuzz

import config_class.mde.premarket as mde_premarket
from data_generation.import_data import download_file_from_s3, get_s3_path
from data_generation.preprocessing import remove_unicode_chars
from config_class.mde.logger import define_logger as logger

# Read the configuration file
home_dir = os.getcwd()
log = logger()
config = mde_premarket.Config(os.path.join(home_dir, "config.yml"))


def extract_page_numbers(pdf_path):
    """
    Extracts list of page number from a PDF document

    params: pdf_path - path to the pdf being processed
    returns: list of page numbers
    """
    try:
        page_numbers = list(range(1, fitz.open(pdf_path).page_count + 1))
        return page_numbers
    except Exception as e:
        log.error(f"Error extracting page numbers from PDF pages: {e}")
        return None


def is_page_searchable(pdf_path, page_number):
    """
    Checks if a pdf page is searchable

    params: pdf_path - path to the pdf file
    params: page_number
    returns
        True if the page is searchable else False
    """
    classification_threshold = int(config["CLASSIFICATION_THRESHOLD"])

    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_number - 1)  # Page numbers start from 0

        image_area = 0.0
        text_area = 0.0

        for b in page.get_text("blocks"):
            r = fitz.Rect(b[:4])
            if "<image:" in b[4]:
                image_area += abs(r)
            else:
                text_area += abs(r)

        if image_area == 0.0 and text_area == 0.0:
            return False  # Page is not searchable/contains significant image objects

        # Avoid division by zero by setting a minimum value for image_area
        image_area = max(image_area, 1.0)

        # Consider a page searchable if it has at least the specified percentage of text
        text_percentage = (text_area / (image_area + text_area)) * 100

        if 100 - text_percentage <= classification_threshold:
            return True  # Page is searchable
        else:
            return False  # Page is not searchable/contains significant image objects
    except Exception as e:
        log.error(f"An error occurred in the is_page_searchable function: {str(e)}")
        return False


def process_pdf_page_with_pytesseract(local_pdf_path, page_number, target_text):
    """
    Gets coordinates for image-based pdf uing tesseract with pymupdf image support
    Args
        local_pdf_path (str) - local path to the downloaded PDF file,
        page_number (int) - the page number of the file to be processed
        target_text (str) - The text of the page whose coordinates is being found
    Returns
        Cordinates of the text
    """
    try:
        with tempfile.TemporaryDirectory() as temp_img_dir:
            # Convert PDF page to image
            image_path = pymupdf_convert_page_to_image(
                pdf_path=local_pdf_path,
                page_number=page_number,
                output_dir=temp_img_dir,
            )

            # Perform OCR on the image
            text_data = pytesseract.image_to_data(
                image_path, output_type=pytesseract.Output.DICT, lang="eng"
            )

            best_match = {"text": "", "bbox": [], "score": 0}
            current_sentence = {"text": "", "bbox": []}

            for i, text in enumerate(text_data["text"]):
                if text_data["block_num"][i] == text_data["block_num"][i - 1]:
                    current_sentence["text"] += text_data["text"][i] + " "
                    current_sentence["bbox"].append(
                        (
                            text_data["left"][i],
                            text_data["top"][i],
                            text_data["width"][i],
                            text_data["height"][i],
                        )
                    )
                else:
                    current_score = fuzz.ratio(
                        current_sentence["text"].strip(), target_text
                    )
                    if current_score > best_match["score"]:
                        best_match["text"] = current_sentence["text"].strip()
                        best_match["bbox"] = current_sentence["bbox"]
                        best_match["score"] = current_score
                    current_sentence = {"text": "", "bbox": [], "block_num": None}

            # Divide each coordinate by 2 to account for zoom factor of 2 in pymupdf extraction
            bbox_normalized = [
                (x // 2, y // 2, w // 2, h // 2) for x, y, w, h in best_match["bbox"]
            ]

            # Calculate coordinates of the entire text
            x_min = min(bbox[0] for bbox in bbox_normalized)
            y_min = min(bbox[1] for bbox in bbox_normalized)
            x_max = max(bbox[0] + bbox[2] for bbox in bbox_normalized)
            y_max = max(bbox[1] + bbox[3] for bbox in bbox_normalized)

            coordinates = {"x1": x_min, "y1": y_min, "x2": x_max, "y2": y_max}

            return coordinates
    except Exception as e:
        # Log the error
        log.error(
            f"An error occurred while process_pdf_page_with_pytesseract: {str(e)}"
        )
        # You might want to handle the error more gracefully based on your use case
        return {"x1": None, "y1": None, "x2": None, "y2": None}


def extract_text_from_pdf(pdf_path, page_number):
    """
    Extracts text from pdf using pymupdf
    params: pdf_path - path to the pdf file
    params: page_number
    returns
        text - text from a PDF page
    """
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_number - 1)  # Page numbers start from 0

        text = page.get_text()

        text = remove_unicode_chars(text)

        return text
    except Exception as e:
        log.error(f"An error occurred while extracting text from PDF: {str(e)}")
        return None


def pymupdf_convert_page_to_image(
    pdf_path, page_number, output_dir, zoom_x=2.0, zoom_y=2.0
):
    """
    Converts a specific page of a PDF to an image.
    Args:
        pdf_filename (str): Path to the PDF file.
        page_number (int): Page number to convert (1-based index).
        output_filename (str): Path to save the output image.
        zoom_x (float): Horizontal zoom factor (default is 2.0).
        zoom_y (float): Vertical zoom factor (default is 2.0).
    Returns
        output path to the page-turned-image
    """
    mat = fitz.Matrix(zoom_x, zoom_y)  # Zoom matrix

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)  # Load the specified page (0-based index)
        pix = page.get_pixmap(matrix=mat)  # Render page to an image
        pix.save(
            os.path.join(output_dir, f"page-{page_number - 1}.png")
        )  # Save image as PNG
        return os.path.join(output_dir, f"page-{page_number - 1}.png")
    except Exception as e:
        log.error(f"Error occured while pymupdf_convert_page_to_image: {e}")
        return None


def extract_text_from_image(pdf_path, page_number, output_dir):
    """
    Extract text from the given image.
    Args:
        image: PIL Image object.
    Returns:
        str: Extracted text from the image.
    """
    image_path = pymupdf_convert_page_to_image(pdf_path, page_number, output_dir)

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)

        text = remove_unicode_chars(text)

        return text
    except Exception as e:
        log.error(f"Error occurred extract_text_from_image: {e}")
        return None


def process_row(row):
    """
    # Start processing inscope dataframe row
    Args:
        row (dict) - data row within the inscope dataframe
    Returns
        list of Langchain documents
    """
    # Initialize document list to store langchain document
    documents = []
    bucket_name = config["BUCKET_NAME"]

    with tempfile.TemporaryDirectory() as from_s3_pdf_temp_dir:
        object_id = row["r_object_id"]
        submission_num = row["folder_id"]
        pdf_path = os.path.join(from_s3_pdf_temp_dir, object_id)

        download_file_from_s3(bucket_name, row, pdf_path)

        s3_path = get_s3_path(bucket_name, row)

        unique_id = f"""{submission_num.lower()}-{object_id.replace('.pdf', '')}"""
        page_number = int(row["page_number"])
        text = row["text_content"]

        searchable = is_page_searchable(pdf_path, page_number)
        page_metadata = {
            "unique_id": unique_id,
            "source_submission_num": submission_num,
            "r_object_id": object_id,
            "r_creation_date": row["r_creation_date"].strftime("%Y-%m-%d %H:%M:%S"),
            "r_modify_date": row["r_modify_date"].strftime("%Y-%m-%d %H:%M:%S"),
            "object_name": row["object_name"],
            "number_of_pages": str(row["number_of_pages"]),
            "submissiontype": row["folder_type"],
            "doc_download_date": row["doc_download_date"].strftime("%Y-%m-%d %H:%M:%S"),
            "page": int(page_number),
            "ocr": "No" if searchable else "Yes",
            "s3_pdf_path": s3_path,
        }
        page_document = Document(page_content=text, metadata=page_metadata)
        documents.append(page_document)

    return documents


def find_text_coordinates_pymupdf(pdf_path, page_number, target_text):
    """
    Finds the coordinates of the bounding box containing the best match
    of the target text in a PDF page using fuzzy matching.

    Args:
    - pdf_path (str): Path to the PDF file.
    - page_number (int): Page number to be processed (0-indexed).
    - target_text (str): The text to search for.

    Returns:
    - Dictionary: Coordinates of the bounding box containing the best match
    of the target text in the format {'x1': value, 'y1': value, 'x2': value, 'y2': value},
    or None if not found.
    """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)  # Corrected page indexing

    # Get the text from the page
    text = page.get_text()

    best_match_score = 0
    best_match_rect = None

    # Find the best match of the target text using fuzzy matching
    for line in nltk.sent_tokenize(text):
        # Calculate fuzzy matching score
        match_score = fuzz.partial_ratio(target_text, line)
        if match_score > best_match_score:
            best_match_score = match_score
            # Extract bounding box coordinates of the best match
            best_match_rect = page.search_for(line)[0]  # Get the first occurrence only

    if best_match_rect:
        return {
            "x1": best_match_rect[0],
            "y1": best_match_rect[1],
            "x2": best_match_rect[2],
            "y2": best_match_rect[3],
        }
    else:
        # If target text not found, return None
        return {"x1": None, "y1": None, "x2": None, "y2": None}


def process_pdf_page_with_pymupdf(metadata, local_path):
    """
    Finds the coordinates of the target text using PyMuPDF.

    Args:
    - metadata (dict): Data dictionary for the the document metadata.
    -local_path (str): Local path where the PDF file has been downloaded.

    Returns:
    - List of tuples: Coordinates of the bounding boxes containing the target text.
    """
    page_number = int(metadata["page"])
    target_text = metadata["document_section"]

    # Find coordinates of the target text in the PDF page
    text_coordinates = find_text_coordinates_pymupdf(
        local_path, page_number - 1, target_text
    )  # Adjust page numbering
    return text_coordinates


def create_langchain_document(group_df):
    """
    Creates a LangChain document - using langchain.schema.Document class to encapsulate the text extracted and associated metadata for every document
    Args:
        group_df (df) - Dataframe of
    Returns:
        documents - List of document text and metadata for the dataframe
    """
    if group_df.empty:
        return []
    documents = []
    for index, data_row in group_df.iterrows():
        row_doc = process_row(data_row)
        documents.extend(row_doc)
    return documents
