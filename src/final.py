import pandas as pd
import os
from ai_analysis import (
    analyze_and_identify_column,
    is_relevant_indicator,
    filter_rows,
    sub_kind_item_mapping,
)
from data_processing import process_data
from utils.hierarchy_utils import assign_hierarchy_order  # Import hàm mới


def final(folder_path):

    # 1. Xử lý dữ liệu ban đầu từ file Excel
    combined_data_cleaned = process_data(folder_path)

    # 2. Phân tích và xác định cột nội dung nhiệm vụ
    identified_column = analyze_and_identify_column(combined_data_cleaned)

    # 3. Đánh số thứ tự cha-con nếu có cột stt
    stt_column = None  # Cột chứa số thứ tự
    if "STT" in combined_data_cleaned.columns:
        stt_column = "STT"
    if "stt" in combined_data_cleaned.columns:
        stt_column = "stt"
    if stt_column is not None:
        combined_data_cleaned = assign_hierarchy_order(
            combined_data_cleaned, stt_column
        )

    if identified_column:
        print(f"Đã xác định được cột cần xử lý: {identified_column}")

        # 4. Sử dụng AI để xác định nhiệm vụ chi thường xuyên đánh trọng số và thêm cột "indicator"
        combined_data_cleaned["indicator"] = combined_data_cleaned[
            identified_column
        ].apply(is_relevant_indicator)

        # 5. Lọc dữ liệu dựa trên trọng số
        filtered_data = filter_rows(combined_data_cleaned)

        # 6. Mapping
        filtered_data["Khoản"] = filtered_data[identified_column].apply(
            sub_kind_item_mapping
        )

        # 7. Đánh lại số thứ tự cha-con
        if stt_column is not None:
            filtered_data = assign_hierarchy_order(filtered_data, stt_column)

        # 8: Lưu kết quả cuối cùng vào file Excel hoặc CSV
        combined_data_cleaned.to_excel("./data/output_indicator.xlsx", index=False)

        filtered_data.to_excel("./data/filtered_output.xlsx", index=False)

        print(filtered_data)
        print(
            "Dữ liệu đã được xử lý, đánh số thứ tự cha-con và lưu vào './data/final_output.xlsx'"
        )
    else:
        print("Không xác định được cột cần xử lý.")


if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "src", "files")
    final(folder_path)
