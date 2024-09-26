# process_and_analyze_data.py

import os
import re
import uuid
import logging
import tempfile
from io import BytesIO, StringIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pytesseract
from PIL import Image

import pandas as pd
import camelot
from werkzeug.utils import secure_filename

from ai_analysis import calculate_relevance_score

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MONEY_PATTERN = re.compile(r"^\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?$")
NUMBER_PATTERN = re.compile(r"^-?\d+[.,]?\d*$")
EXCLUDE_KEYWORDS = {
    "unnamed",
    "phụ biểu",
    "tạm tính",
    "xã",
    "huyện",
    "tỉnh",
    "quận",
    "tổng cộng",
    "stt",
}
KEYWORDS_TO_EXCLUDE = {
    "chỉ tiêu",
    "nội dung",
    "nhiệm vụ",
    "tổng số",
    "dự toán chi",
    "chỉ tiêu xác định dự toán",
    "gồm",
    "gồm:",
    "nhiệm vụ ctx"
}
PREFIXES_TO_EXCLUDE = {"đơn vị tính:"}
ALLOWED_EXTENSIONS = {"xlsx", "xls", "pdf", "png", "jpg", "jpeg"}
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_THREADS", 10))

START_WORDS_TO_EXCLUDE = {
    "nguyễn",
    "trần",
    "lê",
    "hoàng",
    "bùi",
    "vũ",
    "hà",
    "lương",
    "phạm",
    "la",
    "lự",
    "bàn",
    "triệu",
    "ma",
    "chu",
    "vi",
    "lý",
    "ngô",
    "đặng",
    "sầm",
    "nông",
    "hứa",
    "đỗ",
    "dương",
    "phùng",
    "trương",
    "vàng",
    "sùng",
    "vương",
    "giàng",
}

START_WORDS_REGEX = re.compile(
    rf"^(?:{'|'.join(map(re.escape, START_WORDS_TO_EXCLUDE))})\b(?:\s+\w+){{1,3}}$",
    re.IGNORECASE,
)


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_text(text):
    """Determine if the text is valid based on predefined criteria."""
    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 5:
        return False

    if MONEY_PATTERN.match(text):
        return False

    if NUMBER_PATTERN.match(text):
        return False

    if text.startswith("-----") or text.startswith(":-----"):
        return False

    if any(text.lower().startswith(keyword) for keyword in EXCLUDE_KEYWORDS):
        return False

    return True


def convert_dataframe_to_markdown(df):
    """
    Convert a DataFrame to a Markdown-formatted string.
    """
    markdown_buffer = StringIO()
    df.to_markdown(markdown_buffer, index=False)
    markdown_content = markdown_buffer.getvalue()
    logger.debug("Converted DataFrame to Markdown.")
    return markdown_content


def find_best_column_in_markdown(md_string):
    """
    Identify the column in the Markdown table with the most valid texts.
    Returns a list of dictionaries with valid texts from the best column.
    Each dictionary contains a unique 'id' and the 'name' of the text.
    """
    column_count = defaultdict(int)
    valid_texts_per_column = defaultdict(list)

    for line in md_string.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            # Split the line into columns and strip whitespace
            items = [item.strip() for item in line.strip("|").split("|")]

            for col_idx, item in enumerate(items):
                if is_valid_text(item):
                    column_count[col_idx] += 1
                    valid_texts_per_column[col_idx].append(
                        {
                            "id": str(
                                uuid.uuid4()
                            ),  # Generate a unique UUID for each text item
                            "name": item,
                        }
                    )

    if not column_count:
        logger.warning("No valid columns found in Markdown.")
        return []

    # Identify the column with the highest count of valid texts
    best_column = max(column_count, key=column_count.get)
    logger.debug(
        f"Best column identified: Column {best_column} with {column_count[best_column]} valid texts."
    )

    return valid_texts_per_column[best_column]


def process_excel_file(file_stream):
    """
    Process an Excel file and extract valid texts from its sheets via Markdown conversion.
    """
    try:
        sheet_dict = pd.read_excel(file_stream, sheet_name=None, dtype=str)
        logger.info("Successfully read Excel file.")
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return [], [f"Error reading Excel file: {str(e)}"]

    valid_texts = []
    errors = []

    for sheet_name, df in sheet_dict.items():
        logger.debug(f"Processing sheet: {sheet_name}")
        df_filled = df.fillna("").ffill(axis=0)
        md_string = convert_dataframe_to_markdown(df_filled)
        texts = find_best_column_in_markdown(md_string)
        valid_texts.extend(texts)

    return valid_texts, errors


def process_pdf_file(file_stream):
    """
    Process a PDF file and extract valid texts from its tables via Markdown conversion.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(file_stream.read())
            tmp_pdf.flush()
            tables = camelot.read_pdf(tmp_pdf.name, pages="all", flavor="stream")
            logger.info(f"Successfully read PDF file with {tables.n} tables found.")
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return [], [f"Error reading PDF file: {str(e)}"]

    if not tables:
        logger.warning("No tables found in PDF file.")
        return [], ["No tables found in PDF file."]

    valid_texts = []
    errors = []

    for table_num, table in enumerate(tables, start=1):
        logger.debug(f"Processing table {table_num} in PDF.")
        df = table.df
        df_filled = df.fillna("").ffill(axis=0)
        md_string = convert_dataframe_to_markdown(df_filled)
        texts = find_best_column_in_markdown(md_string)
        if texts:
            valid_texts.extend(texts)
        else:
            error_msg = f"No valid texts found in table {table_num}."
            logger.warning(error_msg)
            errors.append(error_msg)

    return valid_texts, errors

def preprocess_image(file_stream):
    """
    Preprocess the image to improve OCR accuracy by removing table lines and adjusting contrast.
    """
    # Load image using OpenCV
    image = np.array(Image.open(file_stream).convert("L"))  # Convert to grayscale

    # Apply binary threshold to make the image black and white
    _, binary_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

    # Define horizontal and vertical kernels to detect lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(
        binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(
        binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Combine both horizontal and vertical lines to form a mask
    mask = cv2.add(horizontal_lines, vertical_lines)

    # Subtract the mask from the original image to remove lines
    processed_image = cv2.subtract(binary_image, mask)

    # Convert back to PIL image for Tesseract OCR
    processed_pil_image = Image.fromarray(cv2.bitwise_not(processed_image))

    return processed_pil_image


def process_image_file(file_stream):
    """
    Process an image file using OCR (Tesseract) and extract valid texts via Markdown conversion.
    """
    try:
        # Load the image using PIL
        # image = Image.open(file_stream)

        # Preprocess the image to improve OCR accuracy
        processed_image = preprocess_image(file_stream)
        
        # Perform OCR on the image
        extracted_text = pytesseract.image_to_string(processed_image, lang="vie", config=r'--oem 3 --psm 12', output_type=pytesseract.Output.STRING)

        # logger.debug(f"Extracted text from image: {extracted_text}")
        
        # Split the extracted text by lines
        lines = extracted_text.split('\n')

        # Initialize an empty list for the cleaned text
        cleaned_lines = []

        # Iterate over lines to remove empty lines and handle lowercase line joining
        previous_line = ""
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # If the line starts with a lowercase letter, append it to the previous line
            if line[0].islower() and previous_line:
                previous_line = previous_line.rstrip() + ' ' + line
            else:
                if previous_line:
                    cleaned_lines.append(previous_line)
                previous_line = line

        # Append the last processed line
        if previous_line:
            cleaned_lines.append(previous_line)

        # Initialize an empty list for the markdown table
        markdown_table = []

        # Iterate over the cleaned lines and process them
        for line in cleaned_lines:
            # Skip empty lines
            if not line:
                continue
            
            # Format the line based on whether it contains hierarchical markers
            if line.startswith('A') or line.startswith('I'):
                # Major section headers
                markdown_table.append(f"| {line} | {'':<30} |")
            elif line.startswith('1') or line.startswith('2'):
                # Subsection with numbers
                markdown_table.append(f"| {line} | {'':<30} |")
            elif line.startswith('+'):
                # Subsection with + symbols
                markdown_table.append(f"| {'':<5} | {line} |")
            else:
                # General text content
                markdown_table.append(f"| {'':<5} | {line} |")
        
        # logger.debug(markdown_table)

        # Convert the table to markdown format
        md_string = "\n".join(markdown_table)

        # Extract the best column with valid texts
        texts = find_best_column_in_markdown(md_string)

        return texts, []

    except Exception as e:
        logger.error(f"Error processing image file: {e}")
        return [], [f"Error processing image file: {str(e)}"]


def filter_texts(texts):
    """
    Filter out texts based on exclusion criteria.
    """
    filtered = []
    for text_obj in texts:
        text = text_obj["name"]
        text_lower = text.lower()

        if any(keyword.lower() == text_lower for keyword in KEYWORDS_TO_EXCLUDE):
            logger.debug(f"Excluding text due to exact keyword match: {text}")
            continue

        if any(text_lower.startswith(prefix.lower()) for prefix in PREFIXES_TO_EXCLUDE):
            logger.debug(f"Excluding text due to prefix match: {text}")
            continue

        if START_WORDS_REGEX.match(text):
            logger.debug(f"Excluding text due to start words rule: {text}")
            continue
        if any(text.lower().startswith(keyword) for keyword in EXCLUDE_KEYWORDS):
            continue

        filtered.append(text_obj)

    logger.info(f"Filtered texts: {len(filtered)} out of {len(texts)}")
    return filtered


def process_files(files):
    """
    Process uploaded files and return a tuple of (valid_texts, errors).
    """
    all_valid_texts = []
    all_errors = []

    if not files:
        logger.warning("No files provided for processing.")
        return all_valid_texts, all_errors

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        future_to_file = {}

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                extension = filename.rsplit(".", 1)[1].lower()
                logger.info(f"Submitting {filename} for processing.")

                try:
                    file_content = file.read()
                    seekable_stream = BytesIO(file_content)

                    if extension in {"xlsx", "xls"}:
                        future = executor.submit(process_excel_file, seekable_stream)
                    elif extension == "pdf":
                        future = executor.submit(process_pdf_file, seekable_stream)
                    elif extension in {"png", "jpg", "jpeg"}:
                        future = executor.submit(process_image_file, seekable_stream)
                    else:
                        # This condition should not occur due to allowed_file check
                        logger.error(f"Unsupported file extension: {extension}")
                        all_errors.append(f"Unsupported file type: {filename}")
                        continue

                    future_to_file[future] = filename
                except Exception as e:
                    logger.error(f"Error preparing file {filename} for processing: {e}")
                    all_errors.append(f"Error preparing file {filename}: {str(e)}")
            else:
                filename = file.filename if file else "No filename"
                logger.error(f"Unsupported or invalid file: {filename}")
                all_errors.append(f"Unsupported or invalid file: {filename}")

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                valid_texts, errors = future.result()
                all_valid_texts.extend(valid_texts)
                if errors:
                    all_errors.extend([f"{filename}: {error}" for error in errors])
                logger.info(f"Completed processing file: {filename}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                all_errors.append(f"Error processing file {filename}: {str(e)}")

    return all_valid_texts, all_errors


def process_files_and_analyze_data(files):
    """
    Process uploaded files and analyze data by calculating relevance scores.
    Returns a tuple of (analyzed_data, errors).
    """
    valid_texts, processing_errors = process_files(files)

    if not valid_texts:
        logger.info("No valid data to analyze after processing files.")
        return [], processing_errors

    # Filter texts based on exclusion criteria
    filtered_texts = filter_texts(valid_texts)
    
    if not filtered_texts:
        logger.info("No data left after filtering texts.")
        return [], processing_errors

    def analyze_item(item):
        """Analyze a single text item to calculate its relevance score."""
        item_name = item.get("name", "")
        if not item_name:
            logger.debug("Skipping item with empty 'name'.")
            return {"id": item.get("id", ""), "name": item_name, "score": 0}

        logger.info(f"Analyzing item: '{item_name}'.")
        try:
            score = calculate_relevance_score(item_name)
            # score = 1
            logger.debug(f"Calculated score for '{item_name}': {score}")
            return {"id": item.get("id", ""), "name": item_name, "score": score}
        except Exception as e:
            logger.error(f"Error calculating relevance score for '{item_name}': {e}")
            return {"id": item.get("id", ""), "name": item_name, "score": 0}

    analyzed_data = []
    analysis_errors = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        future_to_item = {
            executor.submit(analyze_item, item): item for item in filtered_texts
        }

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                analyzed_data.append(result)
            except Exception as e:
                logger.error(f"Error analyzing item '{item.get('name', '')}': {e}")
                analysis_errors.append(
                    f"Error analyzing item '{item.get('name', '')}': {str(e)}"
                )

    total_errors = processing_errors + analysis_errors
    logger.info(
        f"Completed analysis for {len(analyzed_data)} items with {len(total_errors)} error(s)."
    )
    return analyzed_data, total_errors
