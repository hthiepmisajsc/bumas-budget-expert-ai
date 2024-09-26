import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_analysis import calculate_relevance_score, analyze_and_identify_column
import os
import re
from pdfminer.high_level import extract_text
import logging

logging.basicConfig(level=logging.DEBUG)
MAX_THREADS = int(os.getenv("MAX_THREADS", 5))

POSSIBLE_COLUMNS = [
    col.strip().lower()
    for col in os.getenv(
        "POSSIBLE_COLUMNS", "nhiệm vụ,nhiệm vụ chi thường xuyên,chỉ tiêu,nội dung"
    ).split(",")
]


def find_task_column(df):
    """
    Xác định cột liên quan đến nhiệm vụ trong DataFrame.
    Nếu không tìm thấy, sử dụng AI để xác định.
    """
    columns_lower = [col.strip().lower() for col in df.columns]
    logging.debug(f"Columns in DataFrame: {columns_lower}")
    for col_name in POSSIBLE_COLUMNS:
        if col_name in columns_lower:
            return df.columns[columns_lower.index(col_name)]

    return analyze_and_identify_column(df)


def calculate_score_multithreaded(tasks):
    """Tính điểm cho các tasks sử dụng đa luồng."""
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        scores = list(executor.map(calculate_relevance_score, tasks))
    return scores


def process_single_sheet(sheet_name, df):
    """Xử lý một sheet và trả về các tasks với điểm số."""
    result_data = []
    logging.debug(df)

    logging.debug(f"Processing sheet '{sheet_name}'")
    # Chuyển đổi tất cả tên cột thành chuỗi để tránh lỗi khi sử dụng startswith
    logging.debug(f"Columns in sheet '{sheet_name}': {df.columns}")
    df.columns = df.columns.map(str)
    logging.debug(f"Final columns in sheet '{sheet_name}': {df.columns}")
    # Loại bỏ các cột 'Unnamed' và đặt lại header nếu cần
    while any(col.startswith("Unnamed") or pd.isna(col) or col.lower() == 'nan' for col in df.columns):
        df.columns = df.iloc[0].map(str)
        logging.debug(df.columns)
        df = df[1:].reset_index(drop=True)
    logging.debug("end while")
    df.replace("N/A", pd.NA, inplace=True)
    df.dropna(how="all", inplace=True)

    logging.debug("end dropna")
    logging.debug(df)
    identified_column = find_task_column(df)
    logging.debug(f"Find column in sheet '{sheet_name}': '{identified_column}'")

    if identified_column and identified_column in df.columns:
        logging.debug(
            f"Identified column in sheet '{sheet_name}': '{identified_column}'"
        )

        # Check if df[identified_column] is a Series
        column_data = df[identified_column]

        if isinstance(column_data, pd.Series):
            tasks = column_data.dropna().astype(str).tolist()
            scores = calculate_score_multithreaded(tasks)

            for task, score in zip(tasks, scores):
                result_data.append({"name": task, "score": score})
        else:
            logging.debug(
                f"Expected a Series for column '{identified_column}' but got {type(column_data)}."
            )
    else:
        logging.debug(
            f"No task column identified in sheet '{sheet_name}' or column not found."
        )

    return result_data


def process_sheets(file):
    """Xử lý file Excel và trả về danh sách tasks."""
    result_data = []

    try:
        df_sheets = pd.read_excel(file, sheet_name=None)
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        return result_data

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [
            executor.submit(process_single_sheet, sheet_name, df)
            for sheet_name, df in df_sheets.items()
        ]

        for future in as_completed(futures):
            try:
                result_data.extend(future.result())
            except Exception as e:
                logging.error(f"Error processing sheet: {e}")

    return result_data


def extract_tasks_from_text(text):
    """Trích xuất tasks từ văn bản."""
    tasks = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            if re.match(r"^[0-9]+\.?\s*", line):
                task = re.sub(r"^[0-9]+\.?\s*", "", line)
                tasks.append(task)
            else:
                tasks.append(line)
    return tasks


def process_pdf(file):
    """Trích xuất văn bản từ PDF và xử lý tasks."""
    result_data = []

    try:
        text = extract_text(file)
    except Exception as e:
        logging.debug(f"Error extracting text from PDF: {e}")
        return result_data

    tasks = extract_tasks_from_text(text)
    scores = calculate_score_multithreaded(tasks)

    for task, score in zip(tasks, scores):
        result_data.append({"name": task, "score": score})

    return result_data


def process_image(file):
    """Trích xuất văn bản từ hình ảnh và xử lý tasks."""
    # Hiện tại chưa triển khai xử lý hình ảnh
    return []
