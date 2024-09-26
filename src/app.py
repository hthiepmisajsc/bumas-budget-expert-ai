from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import redis
import json
import logging
from predict import estimate_data_predict, parent_predict
from process_and_analyze_data import process_files_and_analyze_data
import os
from functools import wraps

from mongo_handler import store_task_data

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Giới hạn kích thước tệp tối đa là 10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"xlsx", "xls", "pdf", "png", "jpg", "jpeg"}
THRESHOLD = int(os.getenv("THRESHOLD", 7))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
API_KEY = os.getenv("API_KEY", "b4d11cc5-826c-48dd-ac2f-b0fa4d4339e2")

# Kết nối tới Redis
redis_client = redis.StrictRedis(
    host=REDIS_HOST, port=6379, db=0, decode_responses=True
)


def allowed_file(filename):
    """Kiểm tra định dạng file cho phép"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_result(status, data=None, message=""):
    """Định nghĩa cấu trúc kết quả trả về"""
    return {"status": status, "data": data, "message": message}


def get_cached_tasks(key):
    """Lấy dữ liệu tasks từ Redis"""
    cached_result = redis_client.get(key)
    return json.loads(cached_result) if cached_result else []


def update_cached_tasks(key, tasks):
    """Cập nhật tasks vào Redis với thời gian hết hạn"""
    redis_client.set(key, json.dumps(tasks), ex=3600)  # Thời gian hết hạn là 1 giờ


def del_cached_tasks(key):
    """Xóa dữ liệu tasks từ Redis"""
    redis_client.delete(key)


def merge_tasks(cached_tasks, incoming_tasks):
    """Gộp incoming_tasks vào cached_tasks, đè thông tin nếu có trùng id hoặc name."""

    if not isinstance(cached_tasks, list):
        raise ValueError("cached_tasks phải là một danh sách.")

    elif not isinstance(incoming_tasks, list):
        raise ValueError("incoming_tasks phải là một danh sách hoặc None.")

    if incoming_tasks is None:
        incoming_tasks = []

    # Tạo một từ điển từ cached_tasks với key là id, nếu không có id thì dùng name
    tasks_dict = {
        task.get("id") or task.get("name"): task
        for task in cached_tasks
        if "id" in task or "name" in task
    }

    # Lặp qua incoming_tasks và cập nhật tasks_dict
    for task in incoming_tasks:
        # Sử dụng id nếu có, nếu không thì dùng name
        task_key = task.get("id") or task.get("name")

        if task_key:
            # Đè thông tin nếu key đã tồn tại, hoặc thêm mới nếu chưa có
            tasks_dict[task_key] = task
            logger.debug(f"Task with key '{task_key}' updated or added.")

    # Trả về danh sách các task đã được gộp
    return list(tasks_dict.values())


def filter_tasks(tasks, threshold=THRESHOLD):
    """Lọc tasks dựa trên ngưỡng điểm số"""
    return [task for task in tasks if task.get("score", 0) >= threshold]


def remove_duplicate_tasks(tasks):
    """
    Loại bỏ các task trùng lặp dựa trên trường 'name'.
    Giữ lại task sau cùng nếu có trùng lặp.

    :param tasks: List of task dictionaries.
    :return: List of unique task dictionaries.
    """
    tasks_dict = {}
    for task in tasks:
        name = task.get("name")
        if name:
            tasks_dict[name] = task  # Ghi đè task với cùng 'name'
    return list(tasks_dict.values())


def api_key_required(f):
    """Decorator kiểm tra API key"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("api-key")
        if api_key != API_KEY:
            return (
                jsonify(
                    create_result("error", message="Unauthorized: Invalid API key")
                ),
                403,
            )
        return f(*args, **kwargs)

    return decorated_function


@app.errorhandler(404)
def not_found(error):
    return jsonify(create_result("error", message="Endpoint not found")), 404


@app.route("/healthy", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/analysis/task", methods=["POST"])
@api_key_required
def analysis():
    """Xử lý phân tích tasks từ file tải lên."""
    try:
        if "file" not in request.files:
            return (
                jsonify(create_result("error", message="No file part in the request")),
                400,
            )
        files = request.files.getlist("file")

        logger.debug(f"Received {len(files)} file(s) for processing.")
        if not files:
            return (
                jsonify(create_result("error", message="No file part in the request")),
                400,
            )

        all_results, errors = process_files_and_analyze_data(files)
   
        if not all_results and errors:
            if errors:
                logger.error(f"No task data found. Errors: {errors}")
                return (
                    jsonify(
                        create_result(
                            "error",
                            data={"errors": errors},
                            message="No task data found",
                        )
                    ),
                    404,
                )
            else:
                logger.error("No task data found and no errors reported.")
                return (
                    jsonify(create_result("error", message="No task data found")),
                    404,
                )

        # Tạo session_key và lưu vào Redis
        session_key = str(uuid.uuid4())
        update_cached_tasks(session_key, all_results)
        # insert data to turning
        store_task_data(all_results)
        
        logger.debug(f"Session key {session_key} generated and tasks cached.")

        return (
            jsonify(
                create_result(
                    "success",
                    data={
                        "session_key": session_key,
                        "tasks": remove_duplicate_tasks(all_results),
                        "errors": errors or None,
                    },
                    message=(
                        "Data processed successfully"
                        if not errors
                        else "Data processed with some errors"
                    ),
                )
            ),
            200,
        )
    except Exception as e:
        logger.exception("Unexpected error during task analysis.")
        return (
            jsonify(create_result("error", message=f"Error processing file: {str(e)}")),
            500,
        )


@app.route("/analysis/hierarchy/<key>", methods=["POST"])
@api_key_required
def analysis_hierarchy(key):
    """Xử lý phân tích hierarchy cho session key"""
    try:
        if key is None:
            return (
                jsonify(create_result("error", message="Session key is required")),
                400,
            )
        if not request.is_json:
            return (
                jsonify(create_result("error", message="Request data is not JSON")),
                400,
            )

        data = request.get_json()
        incoming_tasks = data.get("tasks", [])

        cached_tasks = get_cached_tasks(key)
        merged_tasks = merge_tasks(cached_tasks, incoming_tasks)
        filtered_tasks = filter_tasks(merged_tasks)
        try:
            processed_tasks = parent_predict(filtered_tasks)
        except Exception as e:
            return (
                jsonify(
                    create_result("error", message=f"Error in hierarchy: {str(e)}")
                ),
                500,
            )

        # del_cached_tasks(key)
        
        # insert data to turning
        store_task_data(processed_tasks)
        
        return (
            jsonify(
                create_result(
                    "success",
                    data={
                        "session_key": key,
                        "tasks": remove_duplicate_tasks(processed_tasks),
                    },
                )
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify(
                create_result("error", message=f"Error processing hierarchy: {str(e)}")
            ),
            500,
        )


@app.route("/analysis/sub_kind_item", methods=["POST"])
@api_key_required
def analysis_sub_kind_item_endpoint():
    """Xử lý phân tích sub-kind item"""
    try:
        if not request.is_json:
            return (
                jsonify(create_result("error", message="Request data is not JSON")),
                400,
            )

        data = request.get_json()
        tasks = data.get("tasks", [])
        sub_kind_items = data.get("sub_kind_items", {})
        sources = data.get("sources", {})
        # sub_kind_items = data.get("sub_kind_items", [])
        # sources = data.get("sources", [])

        try:
            processed_tasks = estimate_data_predict(
                tasks,
                sub_kind_items.get("name", []),
                sources.get("name", []),
                sub_kind_items.get("info", []),
            )
            
            # insert data to turning
            store_task_data(processed_tasks)
        
            # processed_tasks = estimate_data_predict(
            #     tasks,
            #     sub_kind_items,
            #     sources,
            #     None,
            # )
        except Exception as e:
            return (
                jsonify(
                    create_result("error", message=f"Error in prediction: {str(e)}")
                ),
                500,
            )
        return (
            jsonify(
                create_result(
                    "success",
                    data={
                        "tasks": remove_duplicate_tasks(processed_tasks),
                    },
                )
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify(
                create_result(
                    "error", message=f"Error processing sub-kind item: {str(e)}"
                )
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
