import pandas as pd
import os
import glob


# Hàm xử lý dữ liệu từ Excel
def process_excel_files(folder_path):
    # Sử dụng glob để lấy tất cả các file xls, xlsx trong thư mục
    excel_files = glob.glob(os.path.join(folder_path, "*.xls*"))
    print("Excel files found:", excel_files)

    # Tạo danh sách để lưu trữ dữ liệu từ các file Excel
    all_data = []

    # Duyệt qua từng file Excel và đọc dữ liệu
    for file in excel_files:
        # Đọc tất cả các sheet trong file Excel
        xls = pd.ExcelFile(file)

        # Chỉ lấy dữ liệu từ sheet đầu tiên
        df = pd.read_excel(xls, sheet_name=0)
        while df.columns.str.contains("Unnamed").any():
            df.columns = df.iloc[
                0
            ]  # Sử dụng dòng đầu tiên của DataFrame làm header mới
            df = df[1:]  # Bỏ dòng đầu tiên vì đã dùng làm header
            df.reset_index(drop=True, inplace=True)  # Reset lại index

        all_data.append(df)

    # Kết hợp tất cả dữ liệu vào một DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Thay thế các giá trị 'N/A' bằng NaN và xử lý dữ liệu rỗng
    combined_data.replace("N/A", pd.NA, inplace=True)
    combined_data_cleaned = combined_data.dropna(how="all")

    return combined_data_cleaned


# Hàm tổng hợp xử lý cả Excel và ảnh
def process_data(folder_path):
    # Xử lý dữ liệu từ Excel
    return process_excel_files(folder_path)


# Gọi hàm để xử lý dữ liệu
if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "src", "files")
    combined_data = process_data(folder_path)
    print("Combined Data:")
    print(combined_data)
