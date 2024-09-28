import pandas as pd
from datetime import timedelta

# 데이터 읽기
df = pd.read_csv('cell_status.csv', parse_dates=['timestamp'])

# 날짜 열 추가 (날짜만 추출)
df['date'] = df['timestamp'].dt.date

# 데이터를 시간순으로 정렬
df = df.sort_values('timestamp')

# 결과를 저장할 리스트
results = []

# 알고리즘별, 날짜별, 구-동별로 그룹화하여 최대 수면 시간 계산
grouped = df.groupby(['algorithm', 'date', 'district_gu', 'district_dong'])

for (algorithm, date, gu, dong), group_df in grouped:
    max_sleep_time = timedelta()
    current_sleep_start = None
    max_sleep_start = None
    max_sleep_end = None

    # 모든 status가 0인 경우 처리
    if all(group_df['status'] == 0):
        max_sleep_start = group_df['timestamp'].iloc[0]
        max_sleep_end = group_df['timestamp'].iloc[-1]
        max_sleep_time = max_sleep_end - max_sleep_start
    # 모든 status가 1인 경우 처리
    elif all(group_df['status'] == 1):
        max_sleep_time = timedelta(0)  # 수면 시간 없음
    else:
        for i, row in group_df.iterrows():
            if row['status'] == 0:
                if current_sleep_start is None:
                    current_sleep_start = row['timestamp']
            else:
                if current_sleep_start is not None:
                    sleep_duration = row['timestamp'] - current_sleep_start
                    if sleep_duration > max_sleep_time:
                        max_sleep_time = sleep_duration
                        max_sleep_start = current_sleep_start
                        max_sleep_end = row['timestamp']
                    current_sleep_start = None

        # 마지막 구간 체크
        if current_sleep_start is not None:
            sleep_duration = group_df['timestamp'].iloc[-1] - current_sleep_start
            if sleep_duration > max_sleep_time:
                max_sleep_time = sleep_duration
                max_sleep_start = current_sleep_start
                max_sleep_end = group_df['timestamp'].iloc[-1]

    # sleep_time을 "시간:분" 형식으로 변환
    hours, remainder = divmod(max_sleep_time.total_seconds(), 3600)
    minutes = remainder // 60
    sleep_time_str = f"{int(hours):02d}:{int(minutes):02d}"

    results.append({
        'date': date,
        'start_time': max_sleep_start if max_sleep_start else pd.NaT,
        'end_time': max_sleep_end if max_sleep_end else pd.NaT,
        'sleep_time': sleep_time_str,
        'gu': gu,
        'dong': dong,
        'algorithm': algorithm,
    })

# 결과를 DataFrame으로 변환하고 CSV로 저장
result_df = pd.DataFrame(results)
result_df.to_csv('sleep_time_results_grouped.csv', index=False)
