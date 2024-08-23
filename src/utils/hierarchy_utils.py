import pandas as pd

def assign_hierarchy_order(df, stt_column='STT', order_column='Order_Path'):
    """
    Hàm để đánh số thứ tự cha-con dựa trên cột STT, theo cấu trúc phân cấp như ảnh ví dụ.
    
    Args:
    - df: DataFrame chứa dữ liệu cần xử lý.
    - stt_column: Tên của cột chứa số thứ tự (mặc định là 'STT').
    - order_column: Tên của cột mới để lưu trữ số thứ tự cha-con (mặc định là 'Order_Path').
    
    Returns:
    - DataFrame với cột order_column được thêm vào.
    """
    df[order_column] = ""
    level_counters = [0] * 10  # 10 cấp: Cấp 1, Cấp 2, ..., Cấp 10
    current_level = -1

    def reset_levels(start_level):
        """Đặt lại giá trị đếm cho các cấp từ start_level trở xuống"""
        for j in range(start_level, len(level_counters)):
            level_counters[j] = 0

    for i, row in df.iterrows():
        stt = str(row[stt_column]).strip() if not pd.isna(row[stt_column]) else ""

        # Xác định cấp dựa trên định dạng STT
        if stt and (stt.isdigit() or stt[0].isdigit()):  # Số hoặc số với dấu chấm (vd: 2, 2.6, 2.6.1)
            parts = stt.split('.')
            level = len(parts) - 1
            level_counters[level] = int(parts[-1])
            reset_levels(level + 1)
            path = '/'.join(str(level_counters[j]) for j in range(level + 1))
            current_level = level
        
        elif stt and stt.isalpha():  # Các ký tự chữ cái (vd: I, II, III)
            level = 0
            level_counters[level] += 1
            reset_levels(level + 1)
            path = str(level_counters[level])
            current_level = level

        elif stt in ['-', '*', 'N/A'] or stt == "":  # Cấp dựa trên cha gần nhất hoặc NaN
            level = current_level + 1
            level_counters[level] += 1
            reset_levels(level + 1)
            path = '/'.join(str(level_counters[j]) for j in range(level + 1))

        else:
            path = ""

        df.at[i, order_column] = '/' + path + '/'

    return df

# Ví dụ sử dụng:
# df = pd.DataFrame({
#     'STT': ['I', '2', '2.6', '2.6.1', None, None, '2.6.2', None, '2.64', None, '*', 'N/A', '-', '2.7', '2.71', '2.72', 'II', '1.6', '1.61', '1.62', '1.63', '1.64', '1.65']
# })

# df = assign_hierarchy_order(df)
# print(df)
