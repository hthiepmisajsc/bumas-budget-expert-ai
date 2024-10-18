# test_processing.py

import os
import sys
import uuid
import json
import pandas as pd
import logging
from io import BytesIO
from werkzeug.datastructures import FileStorage
from process_and_analyze_data import process_files_and_analyze_data, allowed_file
from datetime import datetime

# Configure logging for the test script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_file_storage(filepath):
    """
    Generate a FileStorage object from a file path to simulate file uploads.
    """
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "rb") as f:
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
    audit_files_dir = os.path.join(os.getcwd(), "src", "audit_files")
    if not os.path.isdir(files_dir):
        logger.error(
            f"Directory '{files_dir}' does not exist. Please create it and add test files."
        )
        sys.exit(1)

    # Gather all supported files in the 'files' directory
    test_files = []
    audit_files = []
    for filename in os.listdir(files_dir):
        filepath = os.path.join(files_dir, filename)
        if os.path.isfile(filepath) and allowed_file(filename):
            file_storage = generate_file_storage(filepath)
            audit_file_storage = generate_file_storage(filepath)
            if file_storage:
                test_files.append(file_storage)
                audit_files.append(audit_file_storage)
        else:
            logger.warning(f"Skipping unsupported or invalid file: {filename}")

    if not test_files:
        logger.error(
            "No valid files found for processing. Ensure 'files' directory contains supported files."
        )
        sys.exit(1)

    logger.info(f"Found {len(test_files)} file(s) for testing.")

    # Process the files
    analyzed_data, errors = process_files_and_analyze_data(test_files)

    # Optionally, save the results to a JSON file
    output_json_file = os.path.join(
        files_dir, f"valid_texts_{int(datetime.now().timestamp())}.json"
    )
    save_texts_to_json(analyzed_data, output_json_file)
    # analyzed_data = [
    #     {
    #         "id": "2dd89b90-cc17-40ea-b5cc-a098f89d2e4a",
    #         "name": "Chi theo Đề án DQTV",
    #         "order": 7,
    #         "file_name": "ANQP",
    #         "score": 10,
    #     }
    #     # Thêm nhiều dữ liệu hơn tại đây
    # ]
    # Print the number of valid texts found
    logger.info(f"Total valid texts found: {len(analyzed_data)}")

    for file_storage in audit_files:
        try:
            # Lấy đường dẫn file từ đối tượng file storage
            file_path = file_storage.filename

            # Tạo luồng đọc file từ file storage
            file_stream = BytesIO(file_storage.read())  # Đọc từ file storage
            xl = pd.read_excel(
                file_stream, sheet_name=None, dtype=str
            )  # Đọc tất cả các sheet trong file Excel

            output_data = {}  # Dùng để lưu dữ liệu kết quả

            # Duyệt qua từng sheet trong file Excel
            for sheet_name, df in xl.items():
                # Đảm bảo cột 'result' có trong DataFrame
                print(df.columns)
                if "result" not in df.columns:
                    logger.warning(
                        f"Sheet {sheet_name} trong {file_path} không có cột 'result'."
                    )
                    continue

                matched_rows = 0
                total_rows = 0

                # Thêm cột 'audit' mới để lưu kết quả kiểm tra
                df["score"] = ""
                df["audit"] = ""

                logger.info(f"Processing sheet '{sheet_name}' in file '{file_path}'...")

                # Xử lý từng hàng
                for idx, row in df.iterrows():
                    result = row["result"]
                    text_found = False
                    # Lặp qua mỗi analyzed_item trong analyzed_data để so sánh
                    for analyzed_item in analyzed_data:
                        name = analyzed_item["name"].strip()
                        score = analyzed_item["score"]
                        # Lặp qua tất cả các giá trị trong hàng của bảng Excel để tìm 'name'
                        for col_value in row:
                            if name in str(
                                col_value
                            ):  # Kiểm tra nếu name khớp với bất kỳ giá trị nào trong hàng
                                # Khi tìm thấy tên khớp, kiểm tra điều kiện score và result
                                result = result if pd.isna(result) else result.strip() 
                                total_rows += 1
                                if score == 10 and (
                                    result == "x"
                                    or result == "nhiệm vụ chi"
                                    or result == "X"
                                    or result == "Nhiệm vụ chi"
                                ):
                                    df.at[idx, "audit"] = "khớp"
                                    matched_rows += 1
                                    text_found = True
                                    df.at[idx, "score"] = score
                                    break
                                # Điều kiện bổ sung: nếu score == 1 và result để trống, cũng coi như khớp
                                elif score == 1 and (
                                    pd.isna(result)
                                    or result == ""
                                    or (
                                        result != "x"
                                        and result != "nhiệm vụ chi"
                                        and result != "X"
                                        and result != "Nhiệm vụ chi"
                                    )
                                ):
                                    df.at[idx, "audit"] = "khớp"
                                    matched_rows += 1
                                    text_found = True
                                    df.at[idx, "score"] = score
                                    break
                                else:
                                    df.at[idx, "audit"] = "không khớp"
                                    df.at[idx, "score"] = score
                                    text_found = True
                                    break
                        if text_found:
                            break

                    # Nếu không tìm thấy khớp, đánh dấu là 'không khớp'
                    # if not text_found and (name or name == ""):
                    #     df.at[idx, "audit"] = "không khớp"
                    #     df.at[idx, "score"] = score

                # Tính toán tỷ lệ khớp: (số dòng 'khớp' / tổng số dòng đã kiểm tra) * 100
                match_percentage = (
                    (matched_rows / total_rows) * 100 if total_rows > 0 else 0
                )

                # Ghi tỷ lệ phần trăm khớp vào ô đầu tiên của cột 'match_percentage'
                df["match_percentage"] = ""
                df.at[0, "match_percentage"] = f"{match_percentage:.2f}%"

                # Thêm dữ liệu đã xử lý vào output_data
                output_data[sheet_name] = df

            # Lưu tất cả các sheet vào một file Excel duy nhất
            output_file = os.path.join(
                audit_files_dir, f"{file_storage.filename}_audit_results.xlsx"
            )
            save_results_to_excel(output_data, output_file)

        except Exception as e:
            logger.error(f"Lỗi khi xử lý file {file_storage.filename}: {str(e)}")
            continue


def save_results_to_excel(output_data, output_file):
    """
    Lưu tất cả các sheet đã phân tích vào một file Excel duy nhất.
    """
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in output_data.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    logger.info(f"Kết quả đã được lưu vào {output_file}")


def save_results_to_csv(output_data, output_file):
    """
    Lưu kết quả đã phân tích vào file CSV.
    """
    # Kết hợp dữ liệu từ các sheet lại thành một DataFrame duy nhất
    combined_df = pd.concat([df for _, df in output_data])

    # Lưu DataFrame kết hợp vào file CSV
    combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    logger.info(f"Kết quả đã được lưu vào {output_file}")


if __name__ == "__main__":
    test_process_files()
