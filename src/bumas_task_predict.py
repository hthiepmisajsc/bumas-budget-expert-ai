from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import logging

# Configure logging
logging.basicConfig(
    level=int(os.getenv("LOG_LEVEL", logging.INFO)),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Khởi tạo mô hình và tokenizer bên ngoài để tái sử dụng
def load_model():
    """
    Hàm để tải mô hình và tokenizer, chỉ khởi tạo một lần.
    """
    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # Lấy thư mục chứa file hiện tại
    model_path = os.path.join(current_dir, "bumas-task-classifier-08")

    # Kiểm tra xem GPU có khả dụng không
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải mô hình đã fine-tune và tokenizer từ thư mục chứa model.safetensors
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Chuyển mô hình sang thiết bị (GPU nếu có)
    model.to(device)

    # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
    model.eval()

    return model, tokenizer, device


def estimase_task_inference(estimase_data):
    if not estimase_data:
        logger.error("No names provided for prediction")
        return []

    # Tải mô hình và tokenizer một lần
    model, tokenizer, device = load_model()

    # Dự đoán cho từng đối tượng trong danh sách
    results = []
    for data in estimase_data:
        result = estimase_task_item_inference(model, tokenizer, device, data["name"])
        results.append({**result, "name": data["name"], "score": result})

    # Giải phóng tài nguyên sau khi hoàn thành dự đoán
    release_model(model)

    return results


def estimase_task_item_inference(model, tokenizer, device, name):
    try:
        if not name:
            logger.error("No names provided for prediction")
            return 1

        # Token hóa văn bản từ trường 'name'
        new_encodings = tokenizer([name], padding=True, return_tensors="pt")

        # Chuyển các tensor đầu vào sang thiết bị (GPU hoặc CPU)
        new_encodings = {key: val.to(device) for key, val in new_encodings.items()}

        # Tắt gradient trong quá trình dự đoán để tăng tốc
        with torch.no_grad():
            outputs = model(**new_encodings)

        # Lấy kết quả phân loại (0: không phải nhiệm vụ dự toán, 1: nhiệm vụ dự toán)
        prediction = torch.argmax(outputs.logits, dim=-1).item()

        return 1 if prediction == 0 else prediction * 10
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        # Đảm bảo giải phóng tài nguyên sau khi sử dụng mô hình
        del model  # Xóa mô hình để giải phóng bộ nhớ
        torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU nếu đang sử dụng GPU


# Hàm để giải phóng mô hình và bộ nhớ
def release_model(model):
    """
    Hàm để giải phóng mô hình khỏi bộ nhớ và giải phóng bộ nhớ GPU.
    """
    try:
        # Xóa mô hình khỏi bộ nhớ
        del model

        # Giải phóng bộ nhớ GPU (nếu đang sử dụng GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Bộ nhớ GPU đã được giải phóng.")
        else:
            print("Không có GPU được sử dụng, không cần giải phóng bộ nhớ.")

    except Exception as e:
        print(f"Error during model release: {str(e)}")


if __name__ == "__main__":
    estimase_names = [
        {"name": "Vật tư văn phòng", "id": "xx1", "order": 1, "file_name": "xa1"},
        {
            "name": "Quỹ lương, phụ cấp và các khoản đóng góp theo lương tính theo số người làm việc thực tế",
            "id": "xx2",
            "order": 2,
            "file_name": "xa2",
        },
        {
            "name": "Tiền ăn (Mức 65.000đ theo Thông tư số 168/2021/TT-BQP)",
            "id": "xx3",
            "order": 3,
            "file_name": "xa3",
        },
    ]
    estimase_task_inference(estimase_names)
