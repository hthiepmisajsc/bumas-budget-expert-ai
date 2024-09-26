from openai import OpenAI
import json
import os
from training_data import estimate_data


# Sử dụng GPT để phân tích
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Chuyển đổi dữ liệu thành cấu trúc JSON
json_file_path = "training_data.json"
converted_data = []
for message in estimate_data["messages"]:
    if message["role"] == "user":
        converted_data.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Bạn là một chuyên gia về lập dự toán chi thường xuyên ngân sách nhà nước Việt Nam.",
                    },
                    {"role": message["role"], "content": message["content"]},
                    {"role": "assistant", "content": message["assistant"]}
                ]
            }
        )

# Lưu dữ liệu đã chuyển đổi thành file JSON
with open(json_file_path, "w", encoding="utf-8") as f:
    for entry in converted_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print(f"Dữ liệu đã được ghi vào {json_file_path}")

# Tải file dữ liệu lên OpenAI
training_file = client.files.create(
    file=open(json_file_path, "rb"), purpose="fine-tune"
)

print(f"Training file ID: {training_file.id}")

# Tạo công việc fine-tuning
fine_tuning_job = client.fine_tuning.jobs.create(
    training_file=training_file.id, model="gpt-4o-mini-2024-07-18"
)

# Theo dõi tiến trình fine-tuning
events = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id=fine_tuning_job.id, limit=10
)
for event in events.data:
    print(event)

# Lấy ID của mô hình đã fine-tune
# fine_tuned_model_id = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)[
#     "fine_tuned_model"
# ]
print(f"Fine-tuned Model ID: {fine_tuning_job.id}")
