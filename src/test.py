# test_processing.py

import os
import sys
import uuid
import json
import logging
from io import BytesIO
from werkzeug.datastructures import FileStorage
from process_and_analyze_data import (
    process_files_and_analyze_data,
    allowed_file
)

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_file_storage(filepath):
    """
    Generate a FileStorage object from a file path to simulate file uploads.
    """
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'rb') as f:
            file_stream = BytesIO(f.read())
            file_storage = FileStorage(stream=file_stream, filename=filename)
            return file_storage
    except Exception as e:
        logger.error(f"Failed to read file {filename}: {e}")
        return None

def save_texts_to_json(texts, output_file):
    """
    Save the list of text dictionaries to a JSON file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved filtered texts to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        
def test_process_files():
    """
    Test the file processing pipeline with files located in the 'files' directory.
    """
    files_dir = os.path.join(os.getcwd(), "src", "files")
    if not os.path.isdir(files_dir):
        logger.error(f"Directory '{files_dir}' does not exist. Please create it and add test files.")
        sys.exit(1)

    # Gather all supported files in the 'files' directory
    test_files = []
    for filename in os.listdir(files_dir):
        filepath = os.path.join(files_dir, filename)
        if os.path.isfile(filepath) and allowed_file(filename):
            file_storage = generate_file_storage(filepath)
            if file_storage:
                test_files.append(file_storage)
        else:
            logger.warning(f"Skipping unsupported or invalid file: {filename}")

    if not test_files:
        logger.error("No valid files found for processing. Ensure 'files' directory contains supported files.")
        sys.exit(1)

    logger.info(f"Found {len(test_files)} file(s) for testing.")

    # Process the files
    analyzed_data, errors = process_files_and_analyze_data(test_files)

    # Optionally, save the results to a JSON file
    output_file = os.path.join(files_dir, "valid_texts.json")
    save_texts_to_json(analyzed_data, output_file)

    # Print the number of valid texts found
    logger.info(f"Total valid texts found: {len(analyzed_data)}")

if __name__ == "__main__":
    test_process_files()
