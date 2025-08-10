"""Module of functions for LLM utility"""

import nltk
import torch
import boto3
from sentence_transformers import util
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockEmbeddings

from config_class.mde.logger import define_logger as logger

# NLTK
# nltk.download("punkt_tab") #for no-internet facing job, this must have been pre-loaded

# Initialize a logger class object
log = logger()

# Loads embedding model
# Initialize the Bedrock Runtime client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-gov-west-1")

embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    endpoint_url="us-gov-west-1",
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-west-1",
)


def setup_llm():
    """
    Sets up the AWS Bedrock LLM class object for Meta's LLama 3B
    """
    llm = BedrockLLM(region_name="us-west-1", model_id="meta.llama3-70b-instruct-v1:0")
    return llm


def prompt_template():
    """
    Prompt template
    returns: prompt
    """
    PROMPT = PromptTemplate(
        template="""
                    <|begin_of_text|>
                    <|start_header_id|>system<|end_header_id|>
                    As an expert in regulatory compliance for medical devices, your task is to meticulously extract the specific reasons for the submission of a 510(k) application to the FDA for pre-market approval. Ensure that the reason is directly quoted from the provided text of the submission without any paraphrasing or rephrasing. If the reason is not explicitly stated in the text, respond with 'Reason not Found'.

                    ### Instructions:
                    1. Read the provided context carefully.
                    2. Identify and quote verbatim the exact reason for the 510(k) submission as stated in the context.
                    3. Do not paraphrase or alter the text in your response.
                    4. If the reason for the submission is not clearly stated in the provided text, your response should be 'Reason not Found'.
                    <|eot_id|>
                    <|start_header_id|>user<|end_header_id|>
                    ### Context:
                    {context}
                    ### Question:
                    {question}

                    Example of a response:
                    "The purpose of this submission is to seek FDA approval for our new medical device, designed to improve patient monitoring through enhanced sensor accuracy."
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["context", "question"],
    )
    return PROMPT


def get_actual_reason_text_from_llm_reason(relevant_text, llm_reason):
    """
    Gets the actual reason by comparing LLM answer with the relevant best-rated document text
    Args:
        relevant_text (str) - RAG best rated document text
        llm_reason (str) - Retrieved LLM-produced reason for submission
    Returns:
        best_match (str) - Best match sentence from the relevant text
    """
    # Splitting the text into sentences
    sentences = nltk.sent_tokenize(relevant_text)

    if not sentences:
        return None

    try:
        reason_embedding = embeddings.embed_query(llm_reason)
        sentence_embeddings = embeddings.embed_documents(sentences)
    except Exception as e:
        log.error(f"Embedding error in getting the actual reason: {e}")
        return None

    # Convert embeddings to PyTorch tensors for cos_sim
    reason_tensor = torch.tensor(reason_embedding).unsqueeze(0)  # shape (1, D)
    sentence_tensor = torch.tensor(sentence_embeddings)  # shape (N, D)

    # Compute cosine similarity
    similarities = util.cos_sim(reason_tensor, sentence_tensor)[0]  # shape (N,)
    best_idx = torch.argmax(similarities).item()

    return sentences[best_idx]
