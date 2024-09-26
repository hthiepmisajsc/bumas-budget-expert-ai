from openai import OpenAI
import os
import json
import logging
from datetime import datetime
from mongo_handler import store_ai_historical_data

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Helper function for making OpenAI API calls
def call_openai_api(
    system_message, user_message, model="gpt-4o-mini", max_tokens=10, temperature=0
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception(f"API call failed: {e}")
        return None


# Analyzes a column based on description
def analyze_column(description):
    system_message = (
        "Bạn là một chuyên gia phân tích dữ liệu về ngân sách nhà nước và quy định lập dự toán chi thường xuyên. "
        "Bạn sẽ phân tích dữ liệu văn bản để xác định xem nó có phải là nhiệm vụ của dự toán chi thường xuyên hay không.\n\n"
        "Chi thường xuyên bao gồm các nhiệm vụ như: chi cho giáo dục, y tế, an sinh xã hội, quốc phòng, an ninh, "
        "chi trả lương cho cán bộ công chức, chi phí hành chính, bảo dưỡng cơ sở vật chất, và các hoạt động thường xuyên khác của các cơ quan, tổ chức công lập.\n\n"
        "Nhiệm vụ của bạn là phân tích đoạn mô tả và xác định xem có phải là nhiệm vụ thuộc chi thường xuyên hay không. "
        "Nếu mô tả khớp với một trong các nhiệm vụ chi thường xuyên, trả về 'true'. Nếu không khớp, trả về 'false'.\n\n"
        "Chỉ trả về đúng từ 'true' hoặc 'false', không thêm bất kỳ nội dung nào khác."
    )
    result = call_openai_api(system_message, description)
    return result.lower() == "true"


# Analyzes and identifies relevant columns from a DataFrame
def analyze_and_identify_column(df):
    column_descriptions = []
    for col in df.columns:
        try:
            values_list = df[col].dropna().head(10).tolist()
            description = (
                f'Tên Cột là "{col}" có các giá trị: {values_list}'
                if values_list
                else ""
            )
            column_descriptions.append(description)
        except Exception as e:
            logger.exception(f"Lỗi khi xử lý cột '{col}': {e}")
            column_descriptions.append("")

    potential_columns = {
        col: analyze_column(desc)
        for col, desc in zip(df.columns, column_descriptions)
        if desc
    }

    for col, analysis in potential_columns.items():
        if analysis:
            logger.debug(f"Cột được xác định là: {col}")
            return col

    logger.debug("Không xác định được cột nào phù hợp.")
    return None


# Calculates relevance score using OpenAI
def calculate_relevance_score(text):
    system_message = (
        "Bạn là một chuyên gia về lập dự toán chi thường xuyên trong ngân sách nhà nước Việt Nam.\n"
        "Nhiệm vụ của bạn là đánh giá xem liệu nội dung sau có phải là một nhiệm vụ của dự toán chi thường xuyên hay không, dựa trên các tiêu chí sau:\n"
        "- Phải thuộc hoạt động thường xuyên và liên tục của đơn vị hành chính sự nghiệp bao gồm: giáo dục, y tế, an sinh xã hội, quốc phòng, an ninh, chi trả lương, chi phí hành chính, bảo dưỡng cơ sở vật chất\n"
        "- Phục vụ cho việc duy trì và vận hành hàng ngày của đơn vị.\n"
        "- Không bao gồm chi phí đầu tư lớn hoặc mua sắm tài sản cố định.\n"
        "Hãy suy nghĩ kỹ và đưa ra một đánh giá bằng số từ 1 đến 10, trong đó:\n"
        "1 - Hoàn toàn không phải là nhiệm vụ chi thường xuyên.\n"
        "10 - Chắc chắn là nhiệm vụ chi thường xuyên.\n"
        "Chỉ trả về một con số từ 1 đến 10. không trả về bất cứ văn bản nào khác"
    )
    result = call_openai_api(
        system_message,
        text,
        model="ft:gpt-4o-mini-2024-07-18:personal:bumas-estimas-score-2:AAXyMp2C",
        max_tokens=1,
    )
    store_ai_historical_data(
        {
            "system_message": system_message,
            "user_message": text,
            "result": result,
            "timestamp": datetime.now().timestamp(),
        }
    )
    try:
        return int(result) if result.isdigit() else 1
    except ValueError:
        logger.exception("Không thể chuyển đổi trọng số.")
        return 1


# Determines if the task is relevant
def is_relevant_task(text):
    system_message = (
        "Bạn là một chuyên gia về lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. "
        "Nhiệm vụ của bạn là đánh giá xem liệu nội dung sau có phải là một nhiệm vụ của dự toán chi thường xuyên hay không?\n"
        "Chỉ trả lời 'có' hoặc 'không'."
    )
    result = call_openai_api(system_message, text)
    return result.lower() == "có"


# Calculates relevance score and provides explanation
def calculate_relevance_score_2(text):
    system_message = (
        "Bạn là một chuyên gia về lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam.\n"
        "Nhiệm vụ của bạn là đánh giá xem liệu nội dung sau có phải là tên của một nhiệm vụ cụ thể trong dự toán chi thường xuyên hay không.\n "
        "Hãy giải thích lý do ngắn gọn nhất tại sao nội dung này có thể hoặc không phải là một nhiệm vụ chi thường xuyên.\n"
        "Sau đó đánh giá con số từ 1 đến 10, trong đó 1 là 'Không phải nhiệm vụ cụ thể', và 10 là 'Chắc chắn là nhiệm vụ cụ thể'."
    )
    try:
        result = call_openai_api(system_message, text, max_tokens=150)
        content = json.loads(result)
        explanation = content.get("explanation", "No explanation provided.")
        score = content.get("score", 1)
    except json.JSONDecodeError:
        explanation = "Unable to parse explanation."
        score = 1
        logger.exception("Unable to parse JSON response.")
    logger.debug(f"Kết quả: {text} => {explanation} score: {score}")
    return score


# Mapping khoản cho từng nhiệm vụ chi thường xuyên
def sub_kind_item_mapping(text, sub_kind_items, info=None):
    sub_kind_item_message = (
        f"Nội dung này có thể nằm trong các khoản: {','.join(sub_kind_items)} \n"
        if sub_kind_items and len(sub_kind_items) > 0
        else ""
    )
    info_message = "\n".join(info) + "\n" if info and len(info) > 0 else ""
    system_message = (
        "Bạn là một chuyên gia lập dự toán chi thường xuyên của ngân sách nhà nước Việt Nam, với kiến thức chuyên sâu về các khoản mục trong mục lục ngân sách.\n"
        "Nhiệm vụ của bạn là phân loại nội dung sau vào khoản phù hợp trong mục lục ngân sách nhà nước.\n"
        f"{info_message}\n"
        f"{sub_kind_item_message}\n"
        "Nếu không có khoản chính xác, hãy sử dụng kiến thức chuyên môn của bạn để xác định khoản gần nhất.\n"
        "Lưu ý: chỉ trả về mã số khoản (ví dụ: '071, 075', '072', '073') hoặc để trống nếu không xác định được.\n"
        "Không trả về bất kỳ văn bản hoặc thông tin bổ sung nào khác ngoài mã số khoản."
    )
    result = call_openai_api(
        system_message,
        text,
        model="ft:gpt-4o-mini-2024-07-18:personal:bumas-estimas-ski:AArgHQX1",
        max_tokens=30,
    )
    store_ai_historical_data(
        {
            "system_message": system_message,
            "user_message": text,
            "result": result,
            "timestamp": datetime.now().timestamp(),
        }
    )
    return result


# Mapping nguồn cho từng nhiệm vụ chi thường xuyên
def source_mapping(text, sources):
    source_message = (
        f"Lưu ý: Chỉ xác định nhiệm vụ thuộc nguồn {','.join(sources)} \n"
        if sources and len(sources) > 0
        else ""
    )
    system_message = (
        "Bạn là một chuyên gia lập dự toán chi thường xuyên của ngân sách nhà nước Việt Nam. \n"
        "Nhiệm vụ của bạn là xác định liệu nội dung sau thuộc nguồn '12' hay '13' trong mục lục ngân sách nhà nước, dựa trên các quy tắc sau:\n"
        "- Nguồn '12': Chi cho các khoản không thường xuyên như chế độ chính sách, tinh giản biên chế, mua sắm, sửa chữa lớn tài sản cố định, đào tạo, bồi dưỡng cán bộ, và các nhiệm vụ ngoài nguồn '13'.\n"
        "- Nguồn '13': Chi cho các khoản thường xuyên như tiền lương, tiền công, phụ cấp lương, các khoản đóng góp theo lương, mua sắm, sửa chữa thường xuyên, và các chi phí thường xuyên theo định mức.\n"
        f"{source_message}"
        "Chỉ trả về mã số '12' hoặc '13', không trả về bất kỳ văn bản nào khác."
    )
    result = call_openai_api(system_message, text, max_tokens=15)

    store_ai_historical_data(
        {
            "system_message": system_message,
            "user_message": text,
            "result": result,
            "timestamp": datetime.now().timestamp(),
        }
    )
    return result


# Identifies the parent task from a list of tasks
def parent_task_mapping(tasks, task_prediction, num_preceding_tasks=10):
    prediction_index = next(
        (i for i, task in enumerate(tasks) if task.get("name", "") == task_prediction),
        None,
    )

    if prediction_index is None:
        logger.debug(f"Nhiệm vụ '{task_prediction}' không tồn tại trong danh sách.")
        return ""

    start_index = (
        0
        if num_preceding_tasks < 10
        else max(0, prediction_index - num_preceding_tasks)
    )

    preceding_tasks = tasks[start_index:prediction_index]

    text = "\n".join(
        f"- {task.get('name', '')}"
        for task in preceding_tasks
        if task.get("name", "") and task.get("name", "") != task_prediction
    )

    system_message = (
        "Bạn là một chuyên gia về lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. "
        "Nhiệm vụ của bạn là xác định nhiệm vụ cha của các nhiệm vụ chi thường xuyên trong danh sách sau:\n"
        f"{text}\n"
        "Nhiệm vụ cha là nhiệm vụ có ý nghĩa bao hàm nhiệm vụ con, hoặc là nhiệm vụ cấp trên của nhiệm vụ con. "
        "Nếu không tìm được cha của nhiệm vụ này, trả về ''. "
        "Chỉ trả về tên nhiệm vụ cha trong danh sách ở trên. Không trả về bất kỳ văn bản nào khác."
    )

    result = call_openai_api(system_message, task_prediction, max_tokens=200)

    is_valid_parent = any(task.get("name", "") == result for task in tasks)

    if not is_valid_parent:
        logger.debug(f"Nhiệm vụ cha '{result}' không tồn tại trong danh sách.")
        return ""

    store_ai_historical_data(
        {
            "system_message": system_message,
            "user_message": task_prediction,
            "result": result,
            "timestamp": datetime.now().timestamp(),
        }
    )
    logger.debug(f"Xác định cha cho '{task_prediction}': '{result}'")
    return result


# Mapping khoản lĩnh vực giáo dục cho từng nhiệm vụ chi thường xuyên
def sub_kind_item_education_mapping(text):
    system_message = (
        "Bạn là một chuyên gia về lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam. "
        "Nhiệm vụ của bạn là xác định liệu nội dung sau thuộc khoản nào trong mục lục ngân sách nhà nước dựa trên các quy tắc sau:\n"
        "- 'Mầm non', 'Mẫu giáo', '24 tháng đến 36 tháng', hoặc các nhiệm vụ phục vụ cấp mầm non, mẫu giáo: chỉ dùng khoản '071'.\n"
        "- 'Tiểu học': chỉ dùng khoản '072'.\n"
        "- 'Trung học', 'THCS': chỉ dùng khoản '073'.\n"
        "- 'Trung cấp', 'Nghề', 'Trung cấp nghề': chỉ dùng khoản '075'.\n"
        "- Nhiệm vụ chứa nghị quyết số '8/2022/NQ ngày 15/7/2022': Khoản '074, 075' \n"
        "- Nhiệm vụ chứa nghị định '116/2016/NĐ-CP ngày 18/7/2016': Khoản '072, 073, 074' \n"
        "Nếu không xác định được cụ thể, trả về tất cả các khoản '071, 072, 073, 074, 075'."
    )
    return call_openai_api(system_message, text, max_tokens=15)


# Filters rows based on a relevance score
def filter_rows(df, min_score=7, keep_parent=False):
    keep_rows = set()
    for index, row in df.iterrows():
        if row.get("score", 0) >= min_score:
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
