"""Module for ARN role assumption"""

import os
import boto3

from config_class.mde.logger import define_logger as logger

# Read the configuration file
home_dir = os.getcwd()
log = logger()

def assume_role_with_arn(role_arn):
    """
    Assume an AWS role using its ARN

    Parameters:
    role_arn (str): The full ARN of the role to assume

    Returns:
    boto3.Session: A new session with the assumed role credentials
    """
    try:
        # Create STS client
        endpoint = ""

        sts_client = boto3.client(
            "sts", endpoint_url=endpoint, region_name="us-west-1"
        )

        # Generate a unique session name using timestamp
        import datetime

        session_name = (
            f'assumed-role-session-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

        # Assume the role
        log.info(f"Attempting to assume role: {role_arn}")

        assumed_role = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
            DurationSeconds=3600,  # 1 hour
        )

        # Create a new session with the temporary credentials
        session = boto3.Session(
            aws_access_key_id=assumed_role["Credentials"]["AccessKeyId"],
            aws_secret_access_key=assumed_role["Credentials"]["SecretAccessKey"],
            aws_session_token=assumed_role["Credentials"]["SessionToken"],
        )

        log.info("Successfully assumed role!")

        return session

    except Exception as e:
        print(f"Error assuming role: {str(e)}")
        raise
