from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from data_processing import process_data
from utils.hierarchy_utils import assign_hierarchy_order

# Tải mô hình và tokenizer từ Hugging Face
model_name = "sail/Sailor-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./models')
# Hàm sử dụng mô hình để tạo phản hồi
def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_p=0.95, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Xử lý dữ liệu ban đầu từ file Excel
def analyze_and_identify_column(df):
    def analyze_column(description):
        prompt = (
            "Bạn là một chuyên gia phân tích dữ liệu với kiến thức sâu rộng về ngân sách nhà nước và các quy định liên quan đến lập dự toán chi thường xuyên.\n"
            "Nhiệm vụ của bạn là phân tích cột dữ liệu được cung cấp và xác định xem cột này có chứa tên các chỉ tiêu, nội dung dự toán, hoặc nhiệm vụ cụ thể trong dự toán chi thường xuyên của ngân sách nhà nước hay không.\n"
            "Một cột phù hợp sẽ chứa các tên nhiệm vụ cụ thể hoặc mục chi tiêu liên quan trực tiếp đến các hoạt động thường xuyên của các cơ quan, tổ chức công lập.\n"
            "Hãy chỉ trả về tên của cột nếu bạn xác định đó là cột chứa nhiệm vụ chi thường xuyên, nếu không thì trả về 'false', và nếu cần thêm thông tin để xác định thì trả về 'none'."
            f"\nMô tả: {description}"
        )
        content = generate_response(prompt)
        print(f'Phân tích cho cột "{description}": {content}')
        return content

    column_descriptions = []
    for col in df.columns:
        description = (
            f'Tên Cột là "{col}" có các giá trị: {df[col].dropna().head(10).tolist()}'
        )
        column_descriptions.append(description)

    potential_columns = {
        col: analyze_column(desc) for col, desc in zip(df.columns, column_descriptions)
    }

    for col, analysis in potential_columns.items():
        if col in analysis:
            print(f"Cột được xác định là: {col}")
            return col

    print("Không xác định được cột nào phù hợp. Cần kiểm tra lại.")
    return None

# Sử dụng GPT để xác định trọng số cho từng chỉ tiêu
def calculate_relevance_score(text):
    prompt = (
        "Bạn là một chuyên gia phân tích dữ liệu ngân sách.\n"
        "Nhiệm vụ của bạn là đánh giá xem liệu nội dung sau có phải là tên của một nhiệm vụ cụ thể trong dự toán chi thường xuyên hay không.\n"
        "Tên nhiệm vụ cụ thể là các khoản chi tiêu liên quan trực tiếp đến hoạt động thường xuyên của cơ quan nhà nước, tổ chức chính trị, tổ chức xã hội, hoặc các đơn vị công lập.\n"
        "Chỉ trả lời bằng một con số từ 1 đến 10, trong đó:\n"
        "1 - Chắc chắn không phải là tên nhiệm vụ cụ thể.\n"
        "10 - Chắc chắn là tên của một nhiệm vụ cụ thể.\n"
        f"Nội dung: {text}"
    )
    content = generate_response(prompt, max_length=10)
    print(f"Xác định trọng số cho: {text} => {content}")
    return int(content.strip())

# Mapping khoản lĩnh vực giáo dục cho từng nhiệm vụ chi thường xuyên
def sub_kind_item_education_mapping(text):
    prompt = (
        "Bạn là một chuyên gia về lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. "
        "Nhiệm vụ của bạn là xác định liệu nội dung sau thuộc khoản nào trong mục lục ngân sách nhà nước dựa trên các quy tắc sau:\n"
        "- 'Mầm non', 'Mẫu giáo', '24 tháng đến 36 tháng', hoặc các nhiệm vụ phục vụ cấp mầm non, mẫu giáo: chỉ dùng khoản '071'.\n"
        "- 'Tiểu học': chỉ dùng khoản '072'.\n"
        "- 'Trung học', 'THCS': chỉ dùng khoản '073'.\n"
        "- 'Trung cấp', 'Nghề', 'Trung cấp nghề': chỉ dùng khoản '075'.\n"
        "Nếu không xác định được cụ thể, trả về tất cả các khoản '071, '072, 073', 074', 075'.\n"
        f"Nội dung: {text}"
    )
    content = generate_response(prompt, max_length=15)
    print(f"Xác định khoản cho: {text} => {content}")
    return content.strip()

# Hàm lọc dữ liệu dựa trên trọng số > 7
def filter_rows(df, min_score=7, keep_parent=False):
    keep_rows = set()
    for index, row in df.iterrows():
        if row["score"] >= min_score:
            keep_rows.add(index)
            if keep_parent:
                order_path = row["Order_Path"]
                path_parts = order_path.strip("/").split("/")
                for i in range(1, len(path_parts)):
                    parent_path = "/" + "/".join(path_parts[:i]) + "/"
                    parent_row = df[df["Order_Path"] == parent_path]
                    if not parent_row.empty:
                        keep_rows.add(parent_row.index[0])

    return df.loc[sorted(keep_rows)]

if __name__ == "__main__":
    print("Chạy AI Analysis với Hugging Face...")
    # Giả định rằng `process_data` và `assign_hierarchy_order` đã được định nghĩa ở nơi khác
    folder_path = os.path.join(os.getcwd(), "src", "files")

    # Xử lý dữ liệu ban đầu từ file Excel
    combined_data_cleaned = process_data(folder_path)
    print(combined_data_cleaned)
    # Đánh số thứ tự cha-con
    combined_data_cleaned = assign_hierarchy_order(combined_data_cleaned)

    # Phân tích và xác định cột nội dung nhiệm vụ
    identified_column = analyze_and_identify_column(combined_data_cleaned)

    if identified_column:
        print(f"Đã xác định được cột cần xử lý: {identified_column}")

        # Sử dụng AI để xác định nhiệm vụ chi thường xuyên đánh trọng số và thêm cột "score"
        combined_data_cleaned["score"] = combined_data_cleaned[identified_column].apply(
            calculate_relevance_score
        )

        # Mapping khoản
        combined_data_cleaned["Khoản"] = combined_data_cleaned[identified_column].apply(
            sub_kind_item_education_mapping
        )

        filtered_data = filter_rows(combined_data_cleaned)
        print("Dữ liệu sau khi lọc:\n\n\n")
        print(filtered_data)
    else:
        print("Không xác định được cột cần xử lý.")
