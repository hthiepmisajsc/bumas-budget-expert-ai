import concurrent.futures
from ai_analysis import sub_kind_item_mapping, source_mapping, parent_task_mapping
import os
import logging

MAX_CONCURRENT_TASKS = int(os.getenv("MAX_THREADS", 10))


def process_task(task, sub_kind_items, sources, info=None):
    """Xử lý sub_kind_item và source cho từng task sử dụng đa luồng."""
    try:
        task["sub_kind_item"] = sub_kind_item_mapping(task["name"], sub_kind_items, info)
        task["source"] = source_mapping(task["name"], sources)
    except Exception as e:
        logging.info(f"Error processing task '{task['name']}': {e}")
        task["sub_kind_item"] = None
        task["source"] = None
    return task


def estimate_data_predict(tasks, sub_kind_items, sources, info=None):
    """Xử lý tất cả các task sử dụng ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        futures = [executor.submit(process_task, task, sub_kind_items, sources, info) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results


def process_parent_task(task, tasks):
    """Xử lý parent task sử dụng đa luồng."""
    try:
        task["parent"] = parent_task_mapping(tasks, task["name"])
    except Exception as e:
        logging.info(f"Error processing task '{task['name']}': {e}")
        task["parent"] = None
    return task


def parent_predict(tasks):
    """Xử lý parent task sử dụng ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        futures = [executor.submit(process_parent_task, task, tasks) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results
