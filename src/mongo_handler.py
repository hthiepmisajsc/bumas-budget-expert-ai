import os
import logging
from pymongo import MongoClient
from datetime import datetime
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MongoDB connection settings
TASKS_COLLECTION = os.getenv("MONGO_TASKS_COLLECTION", "tasks")
AI_HISTORICAL_COLLECTION = os.getenv("MONGO_AI_HISTORICAL_COLLECTION", "ai_historicals")

# Get credentials from environment variables
MONGO_USERNAME = urllib.parse.quote_plus(os.getenv("MONGO_USERNAME", ""))
MONGO_PASSWORD = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD", ""))
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", "27017")
MONGO_DB = os.getenv("MONGO_DB", "analysis_db")

# Construct the MongoDB URI with authentication
if MONGO_USERNAME and MONGO_PASSWORD:
    MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"
else:
    MONGO_URI = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"

# Create a MongoDB client
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

task_collection = db[TASKS_COLLECTION]
ai_historical_collection = db[AI_HISTORICAL_COLLECTION]


def store_task_data(tasks):
    """
    Stores task data into MongoDB.

    :param tasks: List of task data dictionaries.
    :return: True if successful, False otherwise.
    """
    try:
        for task_data in tasks:
            task = task_collection.find_one({"id": task_data["id"]})
            if not task:
                task_data["timestamp"] = datetime.now().timestamp()
                result = task_collection.insert_one(task_data)
                logger.debug(
                    f"Task {task_data['id']} data stored with ID: {result.inserted_id}"
                )
            else:
                updated_data = {}
                if task_data["name"]:
                    updated_data.update({"correction_name": task_data["name"]})
                if task_data["score"]:
                    updated_data.update({"correction_score": task_data["score"]})
                if task_data["sub_kind_item"]:
                    updated_data.update(
                        {"correction_sub_kind_item": task_data["sub_kind_item"]}
                    )
                if task_data["source"]:
                    updated_data.update({"correction_source": task_data["source"]})
                if task_data["parent"]:
                    updated_data.update({"correction_parent": task_data["parent"]})
                if updated_data:
                    updated_data["modified_timestamp"] = datetime.now().timestamp()
                    result = task_collection.update_one(
                        {"id": task_data["id"]}, {"$set": updated_data}
                    )
                    logger.debug(f"Task {task_data['id']} data updated.")
        return True
    except Exception:
        logger.exception("Error storing task data in MongoDB")
        return False


def store_ai_historical_data(ai_historical_data):
    """
    Stores AI historical data into MongoDB.

    :param ai_historical_data: Processed AI historical data.
    :return: True if successful, False otherwise.
    """
    try:
        result = ai_historical_collection.insert_one(ai_historical_data)
        logger.debug(f"AI historical data stored with ID: {result.inserted_id}")
        return True
    except Exception:
        logger.exception("Error storing AI historical data in MongoDB")
        return False
