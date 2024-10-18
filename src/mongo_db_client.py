import os
import logging
from pymongo import MongoClient
from datetime import datetime
from typing import List, Dict
import copy

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MongoDBClient:
    def __init__(self):
        """
        Khởi tạo MongoDBClient, quản lý kết nối và các collection.
        """
        # Lấy thông tin kết nối từ biến môi trường
        self.MONGO_USERNAME = os.getenv("MONGO_USERNAME", "")
        self.MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "")
        self.MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
        self.MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
        self.MONGO_DB = os.getenv("MONGO_DB", "analysis_db")
        self.MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")  # Database để xác thực

        self.client = None
        self.db = None
        self.task_collection = None
        self.ai_historical_collection = None

    def connect(self):
        """
        Kết nối đến MongoDB và thiết lập các collection.
        """
        if not self.client:
            try:
                if self.MONGO_USERNAME and self.MONGO_PASSWORD:
                    self.client = MongoClient(
                        host=self.MONGO_HOST,
                        port=self.MONGO_PORT,
                        username=self.MONGO_USERNAME,
                        password=self.MONGO_PASSWORD,
                        authSource=self.MONGO_AUTH_DB,
                    )
                else:
                    self.client = MongoClient(
                        host=self.MONGO_HOST, port=self.MONGO_PORT
                    )

                self.db = self.client[self.MONGO_DB]
                self.task_collection = self.db["tasks"]
                self.ai_historical_collection = self.db["ai_historicals"]

                logger.info("Connected to MongoDB successfully.")
            except Exception as e:
                logger.exception("Error connecting to MongoDB")
                raise e

    def close(self):
        """
        Đóng kết nối MongoDB.
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def insert_task_data(self, tasks: List[Dict]):
        """
        Chèn dữ liệu task vào MongoDB.

        :param task_data: Dữ liệu task cần chèn.
        :return: True nếu thành công, False nếu có lỗi.
        """
        try:
            if tasks and len(tasks) > 0:
                for task_data in copy.deepcopy(tasks):
                    if task_data:
                        task_data["timestamp"] = datetime.now().timestamp()
                        result = self.task_collection.update_one(
                            {"name": task_data["name"]},
                            {"$set": task_data},
                            upsert=True,
                        )
                        logger.debug(
                            f"Task {task_data['name']} stored with ID: {result.upserted_id}"
                        )
            return True
        except Exception as e:
            logger.exception("Error inserting task data")
            return False

    def update_task_data(
        self, tasks: List[Dict], update_fields: List[str], correction_fields: List[str]
    ):
        """
        Cập nhật dữ liệu task trong MongoDB.
        """
        try:
            if tasks and len(tasks) > 0:
                for task_data in copy.deepcopy(tasks):
                    if task_data:
                        updated_data = {}
                        if correction_fields and len(correction_fields) > 0:
                            for field in correction_fields:
                                if task_data.get(field, ""):
                                    updated_data[f"correction_{field}"] = task_data[
                                        field
                                    ]
                        if update_fields and len(update_fields) > 0:
                            for field in update_fields:
                                if task_data.get(field, ""):
                                    updated_data[field] = task_data[field]
                        if updated_data:
                            updated_data["modified_timestamp"] = (
                                datetime.now().timestamp()
                            )
                            result = self.task_collection.update_one(
                                {"name": task_data["name"]}, {"$set": updated_data}
                            )
                            logger.debug(f"Task {task_data['name']} updated.")
            return True
        except Exception as e:
            logger.exception("Error updating task data")
            return False

    def store_ai_historical_data(self, ai_historical_data):
        """
        Lưu dữ liệu lịch sử AI vào MongoDB.

        :param ai_historical_data: Dữ liệu lịch sử AI đã được xử lý.
        :return: True nếu thành công, False nếu có lỗi.
        """
        try:
            if ai_historical_data:
                result = self.ai_historical_collection.insert_one(ai_historical_data)
                logger.debug(f"AI historical data stored with ID: {result.inserted_id}")
            return True
        except Exception as e:
            logger.exception("Error storing AI historical data")
            return False
