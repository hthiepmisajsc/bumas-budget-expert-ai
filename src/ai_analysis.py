from openai import OpenAI
from data_processing import process_data
from utils.hierarchy_utils import assign_hierarchy_order
import os
import json

# Sử dụng GPT để phân tích và xác định cột
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Xử lý dữ liệu ban đầu từ file Excel
def analyze_and_identify_column(df):
    def analyze_column(description):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là một chuyên gia phân tích dữ liệu với kiến thức sâu rộng về ngân sách nhà nước và các quy định liên quan đến lập dự toán chi thường xuyên.\n"
                        "Nhiệm vụ của bạn là phân tích cột dữ liệu được cung cấp và xác định xem cột này có chứa tên các chỉ tiêu, nội dung dự toán, hoặc nhiệm vụ cụ thể trong dự toán chi thường xuyên của ngân sách nhà nước hay không.\n"
                        "Một cột phù hợp sẽ chứa các tên nhiệm vụ cụ thể hoặc mục chi tiêu liên quan trực tiếp đến các hoạt động thường xuyên của các cơ quan, tổ chức công lập.\n"
                        "Các nhiệm vụ chi thường xuyên bao gồm: giáo dục, y tế, an sinh xã hội, quốc phòng, an ninh, chi trả lương, chi phí hành chính, bảo dưỡng cơ sở vật chất, và các hoạt động khác cần thiết cho việc duy trì hoạt động bình thường của các cơ quan nhà nước.\n"
                        "Hãy chỉ trả về tên của cột nếu bạn xác định đó là cột chứa nhiệm vụ chi thường xuyên, nếu không thì trả về 'false', và nếu cần thêm thông tin để xác định thì trả về 'none'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{description}",
                },
            ],
            max_tokens=10,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        content = response.choices[0].message.content.strip()
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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia phân tích dữ liệu ngân sách.\n"
                    "Nhiệm vụ của bạn là đánh giá xem liệu nội dung sau có phải là tên của một nhiệm vụ cụ thể trong dự toán chi thường xuyên hay không.\n"
                    "Tên nhiệm vụ cụ thể là các khoản chi tiêu liên quan trực tiếp đến hoạt động thường xuyên của cơ quan nhà nước, tổ chức chính trị, tổ chức xã hội, hoặc các đơn vị công lập.\n"
                    "Tên địa phương, phòng ban hoặc cột tính tổng hợp, tính toán trung bình không phải là tên nhiệm vụ cụ thể.\n "
                    "Nếu chỉ tiêu nào không rõ ràng, hãy thêm vào sự hiểu biết của bạn để dự đoán xem nó có phải là tên nhiệm vụ cụ thể hay không.\n"
                    "Hãy suy nghĩ và đánh giá cẩn thận và chỉ trả lời bằng một con số từ 1 đến 10, trong đó:\n"
                    "1 - Chắc chắn không phải là tên nhiệm vụ cụ thể.\n"
                    "10 - Chắc chắn là tên của một nhiệm vụ cụ thể."
                ),
            },
            {
                "role": "user",
                "content": (f"{text}\n\n"),
            },
        ],
        max_tokens=1,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    content = response.choices[0].message.content.strip()
    print(f"Xác định trọng số cho: {text} => {content}")
    return int(content)


def is_relevant_task(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia phân tích dữ liệu ngân sách. "
                    "Nhiệm vụ của bạn là đánh giá xem liệu nội dung sau có phải là tên của một nhiệm vụ cụ thể trong dự toán chi thường xuyên hay không. "
                    "Tên nhiệm vụ cụ thể là các khoản chi tiêu liên quan trực tiếp đến hoạt động thường xuyên của cơ quan nhà nước, tổ chức chính trị, tổ chức xã hội, hoặc các đơn vị công lập.\n"
                    "Tên địa phương, phòng ban hoặc cột tính tổng hợp, tính toán trung bình không phải là tên nhiệm vụ cụ thể.\n "
                    "Các nội dung liên quan đến 'nguồn thu', 'cân đối vào chi thường xuyên', hoặc các hoạt động tổng hợp, tính toán sẽ không phải là nhiệm vụ cụ thể.\n"
                    "Chỉ trả lời 'Có' nếu nội dung là nhiệm vụ cụ thể, và 'Không' nếu không phải."
                ),
            },
            {
                "role": "user",
                "content": f'Nội dung: "{text}"\n\n',
            },
        ],
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
    )
    content = response.choices[0].message.content.strip().lower()
    print(f"Xác định nhiệm vụ cho: {text} => {content}")
    return content == "có"


def calculate_relevance_score_2(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia phân tích ngân sách.\n"
                    "Nhiệm vụ của bạn là đánh giá xem liệu nội dung sau có phải là tên của một nhiệm vụ cụ thể trong dự toán chi thường xuyên hay không.\n "
                    "Hãy giải thích lý do ngắn gọn nhất tại sao nội dung này có thể hoặc không phải là một nhiệm vụ chi thường xuyên\n"
                    "Sau đó đánh giá con số từ 1 đến 10, trong đó 1 là 'Không phải nhiệm vụ cụ thể', và 10 là 'Chắc chắn là nhiệm vụ cụ thể'.\n"
                ),
            },
            {
                "role": "user",
                "content": f'Nội dung: "{text}" return a JSON with "explanation" and "score"\n\n',
            },
        ],
        max_tokens=150,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
    )
    try:
        content = json.loads(response.choices[0].message.content.strip())
        explanation = content.get("explanation", "No explanation provided.")
        score = content.get("score", -1)
    except json.JSONDecodeError:
        explanation = "Unable to parse explanation."
        score = -1
    print(f"Kết quả: {text} => {explanation} score: {score}")
    return score


# Mapping khoản lĩnh vực giáo dục cho từng nhiệm vụ chi thường xuyên
def sub_kind_item_education_mapping(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia về lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. "
                    "Nhiệm vụ của bạn là xác định liệu nội dung sau thuộc khoản nào trong mục lục ngân sách nhà nước dựa trên các quy tắc sau:\n"
                    "- 'Mầm non', 'Mẫu giáo', '24 tháng đến 36 tháng', hoặc các nhiệm vụ phục vụ cấp mầm non, mẫu giáo: chỉ dùng khoản '071'.\n"
                    "- 'Tiểu học': chỉ dùng khoản '072'.\n"
                    "- 'Trung học', 'THCS': chỉ dùng khoản '073'.\n"
                    "- 'Trung cấp', 'Nghề', 'Trung cấp nghề': chỉ dùng khoản '075'.\n"
                    "- Nhiệm vụ chứa nghị quyết, nghị định, thông tư số '29/2020/NQ' ngày '04/12/2020', '81/2021/NĐ' ngày '27/8/2021', '57/2017/NĐ' ngày '09/5/2017', '42/2013/TTLT', '28/2020/NQ' ngày '04/12/2020', '28/2012/NĐ' ngày '10/4/2012': Khoản '071, 072, 073, 074, 075'\n"
                    "- Nhiệm vụ chứa nghị định, nghị quyết số '105/2020/NĐ' ngày '08/9/2020','23/2022/NQ' ngày '07/12/2022': Khoản '071' \n"
                    "- Nhiệm vụ chứa nghị quyết số '8/2022/NQ ngày 15/7/2022': Khoản '074, 075' \n"
                    "- Nhiệm vụ chứa nghị quyết số '09/2022/NQ ngày 15/7/2022': Khoản '071, 072, 073' \n"
                    "- Nhiệm vụ chứa nghị định '116/2016/NĐ-CP ngày 18/7/2016': Khoản '072, 073, 074' \n"
                    "Nếu không xác định được cụ thể, trả về tất cả các khoản '071, '072, 073', 074', 075'.\n"
                    "Chỉ trả về mã số các khoản như '071, 072, 073'. Không trả về bất kỳ văn bản nào khác."
                ),
            },
            {
                "role": "user",
                "content": (f"{text}"),
            },
        ],
        max_tokens=15,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    content = response.choices[0].message.content.strip()
    print(f"Xác định khoản cho: {text} => {content}")
    return content


# Lọc dữ liệu dựa trên trọng số > 7
def filter_rows(df, min_score=7, keep_parent=False):
    keep_rows = set()
    for index, row in df.iterrows():

        if (
            sum(
                [
                    row["score"] >= min_score,
                    row["is_task"] == True,
                    row["score_2"] >= min_score,
                ]
            )
            >= 2
        ):
            # Giữ lại dòng này
            keep_rows.add(index)
            # Giữ lại tất cả các dòng cha
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
    folder_path = os.path.join(os.getcwd(), "src", "files")

    # Xử lý dữ liệu ban đầu từ file Excel
    combined_data_cleaned = process_data(folder_path)

    # Đánh số thứ tự cha-con
    combined_data_cleaned = assign_hierarchy_order(combined_data_cleaned)

    # Phân tích và xác định cột nội dung nhiệm vụ
    columns_lower = [col.lower() for col in combined_data_cleaned.columns]
    if "nhiệm vụ" in columns_lower:
        identified_column = combined_data_cleaned.columns[columns_lower.index("nhiệm vụ")]
    elif "nhiệm vụ chi thường xuyên" in columns_lower:
        identified_column = combined_data_cleaned.columns[columns_lower.index("nhiệm vụ chi thường xuyên")]
    elif "chỉ tiêu" in columns_lower:
        identified_column = combined_data_cleaned.columns[columns_lower.index("chỉ tiêu")]
    elif "nội dung" in columns_lower:
        identified_column = combined_data_cleaned.columns[columns_lower.index("nội dung")]
    else:
        identified_column = analyze_and_identify_column(combined_data_cleaned)

    if identified_column:
        print(f"Đã xác định được cột cần xử lý: {identified_column}")

        # Sử dụng AI để xác định nhiệm vụ chi thường xuyên đánh trọng số và thêm cột "score"
        combined_data_cleaned["score"] = combined_data_cleaned[identified_column].apply(
            calculate_relevance_score
        )

        # Mapping khoản
        print(
            "\n\n\nMapping khoản lĩnh vực giáo dục cho từng nhiệm vụ chi thường xuyên"
        )
        combined_data_cleaned["Khoản"] = combined_data_cleaned[identified_column].apply(
            sub_kind_item_education_mapping
        )

        filtered_data = filter_rows(combined_data_cleaned)
        print("Dữ liệu sau khi lọc:\n\n\n")
        print(filtered_data)
    else:
        print("Không xác định được cột cần xử lý.")
