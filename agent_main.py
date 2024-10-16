# 1. Standard Library Modules
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

# 2. External Library Modules
import spacy

# 3. User-Defined / Project-Specific Modules
from resources.models import SpacyNER
from utils.file_utils import log_restart_count, generate_ddx_id

# 4. Type Hinting / Typing Module
from typing import Tuple, List, Dict, Union

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

        # Load spaCy model only once in the agent_main.py
        start_time = time.time()
        self.nlp = spacy.load("en_core_web_trf")
        logging.info("SpaCy model loaded successfully.")
        end_time = time.time()
        logging.info(f"========== SpaCy model loading time: {end_time - start_time:.2f} seconds =====")

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

    async def hierarchical_model(self):
        # Placeholder for hierarchical model logic
        await asyncio.sleep(1)  # Simulate asynchronous task
        logging.info("Hierarchical model processed.")
        return "Hierarchical Model Result"

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

                # Static input text for demonstration
                input_text = """acme@drio.ai, mobile: 6693225487, SSN: 124-55-8974,
                visa : 1458-9989-6287-6582,
                Acme Inc,
                Address: 54 Clydelle ave San Jose 95124, CA
                The order was placed under PO # 12345, and the billing address is 456 Elm St, New York, NY 10001.
                The customer phone number is 555-678-9101."""

                # Run spaCy model and Hierarchical model concurrently
                spacy_output, hierarchical_output = await asyncio.gather(
                    self.spacy_model(input_text),
                    self.hierarchical_model()
                )

                final_output = {
                    "input_text": input_text,
                    "spacy_output": spacy_output,
                    "hierarchical_output": hierarchical_output
                }

                logging.info(f"Final Output: {json.dumps(final_output, indent=4)}")

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
