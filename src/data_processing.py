import cv2
import pandas as pd
import os
import glob
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
os.environ["TESSDATA_PREFIX"] = r"/usr/share/tesseract-ocr/4.00/tessdata/"


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


def preprocess_image(image_path):
    """Loại bỏ các đường kẻ và cải thiện hình ảnh để OCR hiệu quả hơn."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng threshold để có hình ảnh nhị phân
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Loại bỏ các đường kẻ ngang
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(
        img_bin, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    for c in cnts:
        cv2.drawContours(img_bin, [c], -1, 0, -1)

    # Loại bỏ các đường kẻ dọc
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(
        img_bin, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )
    cnts = cv2.findContours(
        detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    for c in cnts:
        cv2.drawContours(img_bin, [c], -1, 0, -1)

    # Đảo lại hình ảnh về định dạng ban đầu
    img_bin = cv2.bitwise_not(img_bin)

    # Chuyển đổi về định dạng ảnh mà pytesseract có thể làm việc
    img = Image.fromarray(img_bin)
    return img


def extract_text_from_image(image_path):
    """Trích xuất văn bản từ ảnh đã xử lý bằng Tesseract OCR."""
    img = preprocess_image(image_path)

    # Sử dụng Tesseract với cấu hình phù hợp để nhận dạng văn bản
    custom_config = (
        r"--oem 3 --psm 6"  # Set the PSM to 6, assuming a single uniform block of text
    )
    text = pytesseract.image_to_string(img, config=custom_config, lang="vie")
    return text


def parse_extracted_text(text):
    """Phân tích và tái cấu trúc văn bản được trích xuất thành một DataFrame có cấu trúc."""
    lines = text.split("\n")

    # Xử lý từng dòng để tách ra thành các cột
    structured_data = []
    for line in lines:
        # Tách các cột dựa trên khoảng cách giữa các từ hoặc ký tự đặc biệt
        columns = line.split()  # Điều chỉnh cách tách nếu cần
        if len(columns) > 1:
            structured_data.append(columns)

    # Chuyển đổi thành DataFrame
    df = pd.DataFrame(structured_data)
    return df


def process_images_to_dataframe(folder_path):
    """Xử lý tất cả ảnh trong thư mục và kết hợp kết quả thành một DataFrame có cấu trúc."""
    image_files = (
        glob.glob(os.path.join(folder_path, "*.png"))
        + glob.glob(os.path.join(folder_path, "*.jpg"))
        + glob.glob(os.path.join(folder_path, "*.jpeg"))
    )

    all_dataframes = []

    for image_file in image_files:
        print(f"Processing file: {image_file}")
        try:
            text = extract_text_from_image(image_file)
            df = parse_extracted_text(text)
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Kết hợp tất cả DataFrame
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    return combined_df


# Hàm tổng hợp xử lý cả Excel và ảnh
def process_data(folder_path):
    # Xử lý dữ liệu từ Excel
    excel_data = process_excel_files(folder_path)
    # excel_data = None
    # Xử lý dữ liệu từ hình ảnh
    # image_data = process_images_to_dataframe(folder_path)
    image_data = None
    # Gộp dữ liệu từ Excel và dữ liệu từ ảnh thành một DataFrame tổng hợp
    if image_data is not None and not image_data.empty and excel_data is not None:
        combined_data = pd.concat([excel_data, image_data], axis=1, ignore_index=False)
    elif image_data is not None and not image_data.empty:
        combined_data = image_data
    else:
        combined_data = excel_data

    return combined_data


# Gọi hàm để xử lý dữ liệu
if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "src", "files")
    combined_data = process_data(folder_path)
    print("Combined Data:")
    print(combined_data)
