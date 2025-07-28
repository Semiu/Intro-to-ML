"""Module of functions for retrieving information for the LLM pipeline"""

import boto3
import numpy as np
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

import config_class.mde.premarket as mde_premarket
from config_class.mde.logger import define_logger as logger
from data_generation.extract_text import create_langchain_document
from information_retrieval.llm_utils import (
    setup_llm,
    embeddings,
    prompt_template,
    get_actual_reason_text_from_llm_reason,
)
from response_production.prompts import llm_qa_template, response_relevance_template

# Initialize a logger class object
log = logger()
# Reads the configuration file
home_dir = os.getcwd()
config = mde_premarket.Config(os.path.join(home_dir, "config.yml"))

vectorstore = Chroma(
    embedding_function=embeddings, collection_name="all_inscope_documents"
)


def ensemble_retrieval(group_df):
    """
    Combines multiple retrieval strategies to improve document search and retrieval performance
    params: group_df - dataframe to be processed
    returns
        ensemble_retriever - retrieval object of weighted combination of lexical and semantic search methods
    """
    # Creates a list of documents compatible with LangChain document processing and retrieval
    docu = create_langchain_document(group_df)

    # Create splitter object instance to split the document to specified chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = parent_splitter.split_documents(docu)

    # Filter documents/chunks based on the condition
    documents = [
        doc
        for doc in docs if doc is not None]
    documents = [
        doc
        for doc in docs
        if "[ ]" not in doc.page_content and "[]" not in doc.page_content]

    # Initializes the bm25 retriever for lexical search (ranks documents based on keyword relevance)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6  # Limits retrieval to top 6 most relevant documents

    # Index documents in Chroma
    vectorstore.add_documents(documents)

    # Initializes vector retriever (Chroma) for semantic search and Limits retrieval to top 6 most relevant documents
    embedding_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Combines both lexical and semantic retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, embedding_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def get_model_response(prompt):
    """
    Gets model response to the prompt
    Parameters:
        prompt (str): The prompt to guide the LLM model.
    Returns:
        str: LLM model's Response to the question asked.
    """
    # Initialize the Bedrock client
    model_id = "meta.llama3-70b-instruct-v1:0"
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-gov-west-1")

    # Wrap the prompt in a JSON object as required by Bedrock's LLama 3 70B or Claude Haiku
    input_payload = {"prompt": prompt, "max_gen_len": 3000}

    try:
        # Send the request to the Bedrock model
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(input_payload),
        )

        # Parse and return the response
        result = response["body"].read().decode("utf-8")
        result_dict = json.loads(result)
        final_response = result_dict["generation"]

        return final_response

    except Exception as e:
        log.error(f"An error {e} occurred while fetching the LLama model's response.")
        return None


def get_response_relevance_score(response, question, page_content):
    """
    Gets a response relevance score using another LLM (Claude Sonnet) as an evaluator.
    Args:
        response (str): The response fetched from the LLama model, regarding reason for submission.
        question (str): The question asked, to which the response was given.
        page_content (str): The document text to provide context for the Q and A.
    Returns:
        relevance_score (int): LLM-assigned score sugesting the relevance of the response to the question given the page_content.
    """
    # Initialize the Bedrock client
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-gov-west-1")

    prompt = response_relevance_template.format(
        response=response, question=question, document=page_content
    )

    # Wrap the prompt in a JSON object as required by Bedrock's LLama 3 70B or Claude Haiku
    input_payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "anthropic_version": "bedrock-2023-05-31",
    }

    try:
        # Send the request to the Bedrock model
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(input_payload),
        )

        # Parse and return the response
        result = response["body"].read().decode("utf-8")
        result_dict = json.loads(result)
        relevance_score_dict = result_dict["content"][0]["text"]
        relevance_score_str = json.loads(relevance_score_dict)
        relevance_score = relevance_score_str["rating"]

        return int(relevance_score)
    except Exception as e:
        log.error(
            f"An error {e} occured in the claude model function of LLM as a judge"
        )
        return None


def retriever_results(group_df):
    """
    Retrieval results
    params: group_df from the full metadata df
    returns: metadata of LLM-validated RAG response to the question
    """
    query = "what is the purpose of the submission?"
    QUESTION_LIST = config["QUESTION_LIST"]

    try:
        # Get the ensemble retriever documents coded for both lexical and semantic search
        retriever = ensemble_retrieval(group_df)
        source_documents = retriever.invoke(query)

        best_score = 0
        best_metadata = {}
        best_doc = None
        
        # Filtered out None if present
        source_documents = [doc for doc in source_documents if doc is not None]

        for doc in source_documents:
            try:
                # Not interested in very short sentences
                if len(doc.page_content.split()) < 5:
                    continue
                for question in QUESTION_LIST:
                    request_prompt = llm_qa_template.format(
                        question=question, context=doc.page_content
                    )
                    # Get the LLM response to the RAG-retrieved response
                    response = get_model_response(request_prompt)
                    if "reason not found" in response.lower() or response is None:
                        continue
                    # Evaluate the LLM response using another LLM as a judge
                    response_score = get_response_relevance_score(
                        response, question, doc.page_content
                    )
                    if response_score is None:
                        continue
    
                    if int(response_score) > best_score:
                        best_score = response_score
                        best_metadata = (
                            doc.metadata.copy()
                        )  # Assuming metadata is a dictionary that can be copied.
                        best_doc = doc
            except Exception as e:
                continue

        best_metadata["relevancy_score"] = np.round((best_score / 5) * 100, 2)

        # Best document to FAISS Vector Store
        faiss_db = FAISS.from_documents([best_doc], embeddings)

        # Setup LLM pipeline
        llm = setup_llm()

        # Prompt
        final_prompt = prompt_template()

        # Creates a retrieval-based question-answering system using FAISS
        retriever_result = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=faiss_db.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": final_prompt},
        )
        # Generate final result
        retrieval_response = retriever_result.invoke(query)
        answer = retrieval_response["result"]

        # Get the actual reason by comparing LLM answer with the relevant best-rated document text
        actual_reason = get_actual_reason_text_from_llm_reason(
            best_doc.page_content, answer
        )

        best_metadata["reason"] = answer
        best_metadata["document_section"] = actual_reason

        return best_metadata
    except Exception as e:
        log.error(f"An error occurred: {str(e)} in the retriever_results function")
        # Return None in case of an error
        return None
