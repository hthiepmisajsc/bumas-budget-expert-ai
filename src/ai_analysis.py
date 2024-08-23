from openai import OpenAI
from data_processing import process_data
from utils.hierarchy_utils import assign_hierarchy_order
import os

# Sử dụng GPT để phân tích và xác định cột
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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
                        "Lưu ý: Các cột chỉ chứa số liệu tổng hợp, mô tả chung hoặc không liên quan trực tiếp đến một nhiệm vụ cụ thể thì không phải là cột nhiệm vụ chi thường xuyên.\n"
                        "Hãy chỉ trả về tên của cột nếu bạn xác định đó là cột chứa nhiệm vụ chi thường xuyên, nếu không thì trả về 'false', và nếu cần thêm thông tin để xác định thì trả về 'none'."
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


def is_relevant_indicator(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia phân tích dữ liệu và am hiểu về các nghiệp vụ lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. \n"
                    "Nhiệm vụ của bạn là xác định liệu nội dung sau có phải là tên của một nhiệm vụ cụ thể trong dự toán chi thường xuyên hay không.\n "
                    # "Lưu ý: Các danh mục rộng như 'Chi thường xuyên', 'Sự nghiệp giáo dục', hay 'CHI CÂN ĐỐI NGÂN SÁCH ĐỊA PHƯƠNG' không phải là tên nhiệm vụ cụ thể.\n"
                    "Tên các địa phương hoặc các phòng ban như 'Trung tâm dạy nghề và giáo dục thường xuyên' cũng không phải là tên của nhiệm vụ cụ thể. \n"
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
                    "\nNếu không thể xác định chính xác, hãy trả về 'none'. "
                    "Mỗi nhiệm vụ có thể áp dụng nhiều khoản."
                    "\nHãy đánh giá cẩn thận và chỉ trả về mã khoản phù hợp với nhiệm vụ ví dụ:\n"
                    "071, 072, 073\n"
                    "071"
                ),
            },
            {
                "role": "user",
                "content": (f'Nhiệm vụ: "{text}"\n\n'),
            },
        ],
        max_tokens=15,
    )
    content = response.choices[0].message.content.strip()
    print(f"Xác định khoản cho: {text} => {content}")
    return content


def filter_rows(df, keep_parent=False):
    keep_rows = set()
    for index, row in df.iterrows():
        if row["indicator"] >= 7:
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
    identified_column = analyze_and_identify_column(combined_data_cleaned)

    if identified_column:
        print(f"Đã xác định được cột cần xử lý: {identified_column}")

        # Sử dụng AI để xác định nhiệm vụ chi thường xuyên đánh trọng số và thêm cột "indicator"
        combined_data_cleaned["indicator"] = combined_data_cleaned[
            identified_column
        ].apply(is_relevant_indicator)

        # Mapping khoản
        combined_data_cleaned["Khoản"] = combined_data_cleaned[identified_column].apply(
            sub_kind_item_mapping
        )

        filtered_data = filter_rows(combined_data_cleaned)
        print("Dữ liệu sau khi lọc:\n\n\n")
        print(filtered_data)
    else:
        print("Không xác định được cột cần xử lý.")
