"""
Module for the queries used in retrieving data from the Aurora PostGreSQL database tables
"""


def get_reason_for_submission_files():
    """
    Returns the query string to select new OIMT documents for reason-for-submission query analysis.
    Params
    ------
    Returns
    -------
    Query string
    """
    query = """SELECT 
        #to fill
        LIMIT 20000;"""
    return query


def get_extracted_text_query(ids):
    """
    Returns the query string to select extracted text for reason-for-submission query analysis.
    Params: ids - a tuple of the document_ids (r_object_id) to fetch the extracted text of initerest.
    ------
    Returns
    -------
    Query string
    """
    query = f"""SELECT * #to fill IN {ids}
    ;"""

    return query
