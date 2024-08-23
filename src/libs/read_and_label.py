import pandas as pd
import os
import glob
from openai import OpenAI

# Định nghĩa đường dẫn đến thư mục "files"
folder_path = os.path.join(os.getcwd(), "src", "files")

# Sử dụng glob để lấy tất cả các file xls, xlsx trong thư mục
excel_files = glob.glob(os.path.join(folder_path, "*.xls*"))
print(excel_files)

# Tạo danh sách để lưu trữ dữ liệu từ các file Excel
all_data = []
# Duyệt qua từng file Excel và đọc dữ liệu
for file in excel_files:
    # Đọc tất cả các sheet trong file Excel
    xls = pd.ExcelFile(file)

    # Chỉ lấy dữ liệu từ sheet đầu tiên
    df = pd.read_excel(xls, sheet_name=0)
    while df.columns.str.contains("Unnamed").any():
        df.columns = df.iloc[0]  # Sử dụng dòng đầu tiên của DataFrame làm header mới
        df = df[1:]  # Bỏ dòng đầu tiên vì đã dùng làm header
        df.reset_index(drop=True, inplace=True)  # Reset lại index

    df["New_Path"] = ""
    # Khởi tạo biến để lưu trữ thông tin đường dẫn
    path_stack = []
    level_counters = [0] * 5  # Khởi tạo bộ đếm cho 5 cấp

    # Hàm để reset các bộ đếm từ cấp cụ thể trở xuống
    def reset_counters_from_level(level):
        for i in range(level, len(level_counters)):
            level_counters[i] = 0

    # Duyệt qua từng hàng trong DataFrame
    for i, row in df.iterrows():
        stt = str(row["STT"]).strip()
        # Bỏ qua các giá trị NaN hoặc trống
        if stt.lower() in ["nan", ""]:
            continue

        if stt.isalpha() and stt.isupper() and len(stt) == 1:  # A, B, C... (Cấp 1)
            level_counters[0] += 1
            reset_counters_from_level(1)
            path_stack = [str(level_counters[0])]
        elif stt.isupper() and len(stt) > 1:  # I, II, III... (Cấp 2)
            level_counters[1] += 1
            reset_counters_from_level(2)
            path_stack = [str(level_counters[0]), str(level_counters[1])]
        elif stt.isdigit():  # 1, 2, 3... (Cấp 3)
            level_counters[2] += 1
            reset_counters_from_level(3)
            path_stack = [
                str(level_counters[0]),
                str(level_counters[1]),
                str(level_counters[2]),
            ]
        elif stt.islower():  # a, b, c... (Cấp 4)
            level_counters[3] += 1
            reset_counters_from_level(4)
            path_stack = [
                str(level_counters[0]),
                str(level_counters[1]),
                str(level_counters[2]),
                str(level_counters[3]),
            ]
        elif stt in ["+", "-", "*"]:  # -, +, * (Cấp 5)
            level_counters[4] += 1
            path_stack.append(str(level_counters[4]))

    # Tạo đường dẫn theo quy tắc
    df.at[i, "New_Path"] = "/" + "/".join(path_stack) + "/"
    all_data.append(df)

# Kết hợp tất cả dữ liệu vào một DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Thay thế các giá trị 'N/A' bằng NaN và xử lý dữ liệu rỗng
combined_data.replace("N/A", pd.NA, inplace=True)
combined_data_cleaned = combined_data.dropna(how="all")

print(combined_data_cleaned)

# Sử dụng GPT để phân tích và xác định cột
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def analyze_and_identify_column(df):
    def analyze_column(description):
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Sử dụng mô hình nhẹ hơn nếu có thể
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là một chuyên gia phân tích dữ liệu. Hãy phân tích dữ liệu sau và xác định cột này có phải là cột chứa tên chỉ tiêu, nội dung dự toán, nhiệm vụ của dự toán chi thường xuyên không?.\n"
                        "Chỉ trả về tên cột nếu bạn xác định được. không phải thì chỉ trả về fasle, cột cần xem xét thêm thì trả về none."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{description}",
                },
            ],
            max_tokens=10,
        )
        content = response.choices[0].message.content.strip()
        print(f'Phân tích cho "{description}": {content}')
        return content

    # Giảm số lượng giá trị gửi lên GPT để phân tích
    column_descriptions = []
    for col in df.columns:
        # Chỉ lấy 3 giá trị đầu tiên để gửi lên GPT
        description = f'Cột "{col}" có các giá trị: {df[col].dropna().head(3).tolist()}'
        column_descriptions.append(description)

    # Gọi API GPT chỉ một lần để phân tích
    potential_columns = {
        col: analyze_column(desc) for col, desc in zip(df.columns, column_descriptions)
    }

    # Xác định cột chứa "chỉ tiêu" hoặc "nội dung dự toán"
    for col, analysis in potential_columns.items():
        if col in analysis:
            print(f"Cột được xác định là: {col}")
            return col

    print("Không xác định được cột nào phù hợp. Cần kiểm tra lại.")
    return None


# Áp dụng hàm để xác định cột mục tiêu
identified_column = analyze_and_identify_column(combined_data_cleaned)

if identified_column:
    # Thực hiện các bước tiếp theo như lọc và làm sạch dữ liệu
    print(f"Đã xác định được cột cần xử lý: {identified_column}")

    def is_relevant_indicator(text):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là một chuyên gia phân tích dữ liệu và am hiểu về các nghiệp vụ lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. \n"
                        "Nhiệm vụ của bạn là xác định liệu nội dung sau có phải là tên của một nhiệm vụ cụ thể trong dự toán chi thường xuyên hay không.\n "
                        "Lưu ý: Các danh mục rộng như 'Chi thường xuyên', 'Sự nghiệp giáo dục', hay 'CHI CÂN ĐỐI NGÂN SÁCH ĐỊA PHƯƠNG' không phải là tên nhiệm vụ cụ thể.\n"
                        "Các nội dung mang tính chất tổng hợp số liệu, tính tổng, tính trung bình, tính phần trăm, không phải là tên nhiệm vụ cụ thể.\n"
                        "Nhiệm vụ dự toán chi thường xuyên là các khoản chi tiêu của ngân sách nhà nước được lập và phê duyệt hàng năm, nhằm đảm bảo các hoạt động thường xuyên của các cơ quan, tổ chức, đơn vị công lập, cũng như các chương trình, dự án cần thiết để duy trì hoạt động bình thường của các cơ quan nhà nước, tổ chức chính trị, tổ chức chính trị - xã hội, và các đơn vị sự nghiệp công lập\n"
                        "Nhiệm vụ chi thường xuyên thường bao gồm các chi phí liên quan đến giáo dục, y tế, an sinh xã hội, quốc phòng, an ninh, chi trả lương cho cán bộ, chi phí hành chính, bảo dưỡng cơ sở vật chất, và các chi phí văn phòng khác​\n"
                        "Hãy đánh giá cẩn thận và chỉ trả lời bằng một con số từ 1 đến 10, trong đó:\n"
                        "1 - Chắc chắn không phải là tên nhiệm vụ cụ thể.\n"
                        "10 - Chắc chắn là tên của một nhiệm vụ cụ thể."
                    ),
                },
                {
                    "role": "user",
                    "content": (f'Nội dung: "{text}"\n\n'),
                },
            ],
            max_tokens=15,
        )
        content = response.choices[0].message.content.strip()
        print(f"Xác định trọng số cho: {text} => {content}")
        return int(content)

    combined_data_cleaned["Trọng số"] = combined_data_cleaned[identified_column].apply(
        is_relevant_indicator
    )

    print(combined_data_cleaned)

    # Lọc bỏ các hàng có trọng số nhỏ hơn 7, trừ khi là tiêu chí cha có tiêu chí con với trọng số >= 7
    def filter_rows(df):
        keep_rows = []
        for index, row in df.iterrows():
            if row["Trọng số"] >= 7:
                keep_rows.append(index)
            # else:
            #     # Kiểm tra xem hàng kế tiếp có phải là tiêu chí con với trọng số >= 7
            #     if index < len(df) - 1:
            #         next_row = df.iloc[index + 1]
            #         if next_row["Trọng số"] >= 7:
            #             keep_rows.append(index)
        return df.loc[keep_rows]

    filtered_data = filter_rows(combined_data_cleaned)

    print("Dữ liệu sau khi lọc", filtered_data)
else:
    print("Không xác định được cột cần xử lý.")


def sub_kind_item_mapping(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia phân tích dữ liệu và am hiểu về các nghiệp vụ lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. \n"
                    "Nhiệm vụ của bạn là xác định liệu nội dung sau thuộc khoản nào trong mục lục ngân sách nhà nước\n "
                    "Nếu tên nhiệm vụ có chứa 'Mầm non', 'Mẫu giáo', '24 tháng đến 36 tháng', hoặc nó phục vụ cho cấp mầm non, mẫu giáo thì thuộc khoản '071'\n"
                    "Nếu tên nhiệm vụ thuộc nghị quyết số '29/2020/NQ' ngày '04/12/2020', Nghị định số '81/2021/NĐ' ngày '27/8/2021', Nghị định số '57/2017/NĐ' ngày '09/5/2017', Thông tư số '42/2013/TTLT', Nghị quyết số '28/2020/NQ' ngày '04/12/2020', Nghị định số '28/2012/NĐ' ngày '10/4/2012' thì thuộc khoản '071', '072', '073', '074', '075'\n"
                    "Nếu tên nhiệm vụ thuộc Nghị định '105/2020/NĐ' ngày '08/9/2020', Nghị quyết số '23/2022/NQ' ngày '07/12/2022' thuộc khoản '071' \n"
                    "Nếu tên nhiệm vụ thuộc Nghị quyết số '8/2022/NQ' ngày '15/7/2022' thì thuộc khoản '074', '075' \n"
                    "Nếu tên nhiệm vụ thuộc Nghị quyết số '09/2022/NQ' ngày '15/7/2022' thì thuộc khoản '071', '072', '073' \n"
                    "Nếu tên nhiệm vụ thuộc nghị định '116/2016/NĐ-CP' ngày '18/7/2016' thì thuộc khoản '072', '073', '074' \n"
                    "Nếu tên nhiệm vụ có chứa 'tiểu học' hoặc nó mục đích phục vụ cho cấp tiểu học thì thuộc khoản '072'\n"
                    "Nếu tên nhiệm vụ có chứa 'trung học', 'THCS' hoặc nó mục đích phục vụ cho cấp trung học thì thuộc khoản '073'\n"
                    "Nếu tên nhiệm vụ có chứa 'trung cấp', 'nghề', 'trung cấp nghề' hoặc nó mục đích phục vụ cho cấp trung cấp nghề thì thuộc khoản '075'\n"
                    "Các khoản chi cho giáo dục là '071', '072', '073', '074', '075' trong đó: \n"
                    "Khoản chi cho y tế là '072' trong đó\n"
                    "Khoản Chi cho văn hóa, thông tin, thể thao là '073' trong đó\n"
                    "Khoản chi cho khoa học và công nghệ là '074' trong đó\n"
                    "Khoản chi cho Hoạt động Kinh tế là '075' trong đó\n"
                    "Quốc phòng và An ninh là '076', '077' trong đó\n"
                    "Chi Bảo đảm Xã hội là khoản '080' trong đó\n"
                    "Chi cho Quản lý Hành chính Nhà nước là khoản '090' trong đó\n"
                    "Chi khác là khoản '100' trong đó\n"
                    "Chi sự nghiệp xã hội là khoản '081' trong đó\n"
                    "Chi sự nghiệp môi trường là khoản '082' trong đó\n"
                    "Chi cho các hoạt động kinh tế khác là khoản '083' trong đó\n"
                    "Chi dự trữ quốc gia là khoản '084' trong đó\n"
                    "Chi trả nợ và viện trợ là khoản '085' trong đó\n"
                    "Nếu nhiệm vụ không xác định được cụ thể mà xác định được thuộc lĩnh vực nào như 'giáo dục', 'quốc phòng', ... thì điền tất cả các khoản thuộc lĩnh vực đó cho nhiệm vụ này"
                    "Mỗi một nhiệm vụ có thể dùng nhiều khoản\n"
                    "Hãy đánh giá cẩn thận và chỉ trả lời là các khoản nào, nếu không chắc chắn thì trả về 'none'.",
                ),
            },
            {
                "role": "user",
                "content": (f'Nội dung: "{text}"\n\n'),
            },
        ],
        max_tokens=15,
    )
    content = response.choices[0].message.content.strip()
    print(f"Xác định khoản cho: {text} => {content}")
    return content