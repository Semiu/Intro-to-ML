"""Module for the text preprocessing functions"""

import re


def remove_unicode_chars(input_str):
    """
    Removes Unicode characters
    params: input_str
    returns: preprocessed string - of removed
    """
    string_encode = input_str.encode("ascii", "ignore")
    string_decode = string_encode.decode()

    string_decode = (
        string_decode.replace("'", " ")
        .replace("\t", " ")
        .replace("\n", " ")
        .replace("\x00", " ")
    )
    string_decode = re.sub(" +", " ", string_decode)

    return string_decode.strip()


def preprocess_text(text):
    """
    Preprocess a given text
    Arg:
        text (str) - any text provided
    returns - processed text where special character and spaces are substituted
    """
    # Remove special characters and spaces and convert to lowercase
    return re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
