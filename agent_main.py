import sys
import os
import signal
import time
import uuid
import socket
from time import sleep
import logging
import json
import asyncio
import spacy
from typing import Tuple, List, Dict, Union
from resources.models import SpacyNER, TopicModelProcessor
from utils.file_utils import log_restart_count, generate_ddx_id
from PDFExtractDemo.pdf_processor import start_pdf_extraction_engine  # Importing your old PDF processing method

LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='/var/log/ddx-agent.out.log', format=LOG_FORMAT, level=logging.INFO)

DDX_ID: str = str(uuid.uuid4())
SERVICE_PROVIDER: str = os.environ.get('SERVICE_PROVIDER') or 'local'
DDX_INSTANCE_NAME: str = os.environ.get('DDX_INSTANCE_NAME') or socket.gethostname()

class DDXAgent:
    def __init__(self) -> None:
        global DDX_ID
        DDX_ID = generate_ddx_id(DDX_ID)
        log_restart_count()

        self.kill_now: bool = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        # Measure the total loading time
        total_start_time = time.time()

        # Measure spaCy model loading time
        spacy_start_time = time.time()
        self.nlp = spacy.load("en_core_web_trf")
        spacy_end_time = time.time()
        spacy_loading_time = spacy_end_time - spacy_start_time
        logging.info(f"========SpaCy model loaded successfully in {spacy_loading_time:.2f} seconds.")

        # Path to the model for BERTopic
        bertopic_start_time = time.time()
        self.model_path = "/docker-entrypoint-ddx.d/ddx_agent/resources/Merged_model_09_26_2024"

        # Initialize the TopicModelProcessor with the model path
        self.topic_processor = TopicModelProcessor(self.model_path)
        bertopic_end_time = time.time()
        bertopic_loading_time = bertopic_end_time - bertopic_start_time
        logging.info(f"=========BERTopic model loaded successfully in {bertopic_loading_time:.2f} seconds.")

        # Calculate the total loading time
        total_end_time = time.time()
        total_loading_time = total_end_time - total_start_time
        logging.info(f"========== Total model loading time: {total_loading_time:.2f} seconds =====")

    def exit_gracefully(self, *args) -> None:
        self.kill_now = True

    # ------------------------------------------------------------------------------
    #                           Spacy model
    # ------------------------------------------------------------------------------

    async def spacy_model(self, input_text: str) -> Dict[str, Union[List[Dict[str, Union[str, int, None]]], str]]:
        
        zip_code_file = "/docker-entrypoint-ddx.d/ddx_agent/resources/ZIP_Locale_Detail.xlsx"
        
        # Pass the loaded spaCy model to SpacyNER
        ner = SpacyNER(zip_code_file, self.nlp)
        
        # Process input text with the spaCy model
        result = ner.process_text(input_text)
        logging.info(f"SpaCy model results: {json.dumps(result, indent=4)}")
        return result

    # ------------------------------------------------------------------------------
    #                           Hierarchical model
    # ------------------------------------------------------------------------------

    async def process_documents(self, input_text: str) -> dict:
        """Process documents by retrieving input text and processing them with BERTopic"""
        df_probs = self.topic_processor.process_text(input_text)
        topic_info_json = self.topic_processor.get_topic_info()

        # Convert DataFrame to a dictionary to ensure JSON serializability
        df_probs_dict = df_probs.to_dict()
        topic_info_json = json.loads(topic_info_json)  # Ensure JSON format

        return {
            "Processed_DataFrame": df_probs_dict,
            "Topic_Information": topic_info_json
        }

    # ------------------------------------------------------------------------------
    #                           Main Task
    # ------------------------------------------------------------------------------

    async def run_synchronous_tasks(self) -> None:
        """
        Processes input text using spaCy and Hierarchical models, then logs the results.
        """
        try:
            while not self.kill_now:
                logging.info("Running synchronous tasks...")

                # Get input from PDF extraction engine (updated method)
                pdf_data = start_pdf_extraction_engine()

                if pdf_data:
                    input_text = pdf_data.get("processed_data")  # Extract input data from the PDF
                    filename = pdf_data.get("filename")
                    
                    # Handle cases where 'actual_output' might not exist
                    actual_output = pdf_data.get("actual_output", None)  # Use None as default if 'actual_output' is missing

                    if input_text:
                        # Run spaCy model and Hierarchical model concurrently
                        spacy_output, hierarchical_output = await asyncio.gather(
                            self.spacy_model(input_text),
                            self.process_documents(input_text)  # Ensure process_documents is awaited
                        )

                        final_output = {
                            "filename": filename,
                            "input_text": input_text,
                            "spacy_output": spacy_output,
                            "hierarchical_output": hierarchical_output
                        }

                        # Properly format the final_output as JSON string
                        formatted_json = json.dumps(final_output, indent=4)
                        logging.info(f"Final Output: {formatted_json}")
                    else:
                        logging.warning("No input text found in PDF data.")

        except Exception as e:
            logging.error(f"An error occurred in run_synchronous_tasks: {e}")

    # ------------------------------------------------------------------------------
    #                           Main Run Loop
    # ------------------------------------------------------------------------------

    async def run(self) -> None:
        """
        Starts the main asynchronous loop that runs the models.
        """
        try:
            await self.run_synchronous_tasks()
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")

if __name__ == '__main__':
    logging.info("Starting Main")
    agent: DDXAgent = DDXAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass
