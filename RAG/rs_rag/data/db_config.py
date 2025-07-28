"""Module for creating the PostgreSQL db connection object"""

import os
import boto3
import json
import psycopg2
from configparser import ConfigParser
from botocore.exceptions import ClientError

import config_class.mde.premarket as mde_premarket
from config_class.mde.logger import define_logger as logger


def config_credentials(filename=f"{os.getcwd()}/credentials.ini", section="aurora"):
    """
    DB configuration method for the dev
    Reads the db params from a git-ignored file
    Returns
        A dictionary of key-value pair for the db parameters
    """
    parser = ConfigParser()
    parser.read(filename)

    # get section, default to postgresql
    param_dict = {}

    # Checks to see if section (postgresql) parser exists
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            param_dict[param[0]] = param[1]

    # Returns an error if a parameter is called that is not listed in the initialization file
    else:
        raise Exception(
            "Section {0} not found in the {1} file".format(section, filename)
        )

    return param_dict


def get_db_parameters():
    """
    Gets the database connection parameters from the AWS Secret manager
    """
    log = logger()
    home_dir = os.getcwd()
    config = mde_premarket.Config(os.path.join(home_dir, "config.yml"))

    secret_name = config["security"]["secret_name"]
    region_name = config["security"]["region_name"]

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name,
        # Commented out before it can succesfully be used through the Aurora-designated Sagemaker notebook instance
        endpoint_url="",
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        log.error(f"Error {e} in retrieving the secret value response")

    # Decrypts secret using the associated KMS key
    secret = get_secret_value_response["SecretString"]

    # Parse the JSON string
    secret_dict = json.loads(secret)

    params = {
        "host": secret_dict["host"],
        "user": secret_dict["username"],
        "password": secret_dict["password"],
        "database": secret_dict["dbname"],
        "port": secret_dict["port"],
    }

    return params


def get_db_connection():
    """
    Gets the Postgres connection object
    """
    log = logger()
    try:
        params = get_db_parameters()  # for preprod
        # params = config_credentials() # for dev

        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**params)

        return conn
    except Exception as e:
        log.error(f"Unable to get the database connection due to {e}")
