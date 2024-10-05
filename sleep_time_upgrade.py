import pandas as pd
import numpy as np

# 데이터 읽기
df = pd.read_csv('cell_status.csv', parse_dates=['timestamp'])

# 날짜 열 추가 (날짜만 추출)
df['date'] = df['timestamp'].dt.date

# 데이터를 시간순으로 정렬
df = df.sort_values('timestamp')

# status 변화 지점 찾기
df['status_change'] = df['status'].diff().ne(0)
df.loc[df.index[0], 'status_change'] = True

def calculate_sleep_time(group):
    change_points = group[group['status_change']]
    
    if len(change_points) <= 1:
        if group['status'].iloc[0] == 0:
            return pd.Series({
                'start_time': group['timestamp'].iloc[0],
                'end_time': group['timestamp'].iloc[-1],
                'sleep_time': (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).total_seconds() / 60
            })
        else:
            return pd.Series({'start_time': pd.NaT, 'end_time': pd.NaT, 'sleep_time': 0})
    
    sleep_periods = change_points['timestamp'].iloc[1:].values - change_points['timestamp'].iloc[:-1].values
    sleep_periods = sleep_periods[::2]  # 짝수 인덱스만 선택 (0 상태 기간)
    
    if change_points['status'].iloc[0] == 0:
        first_period = change_points['timestamp'].iloc[1] - group['timestamp'].iloc[0]
        sleep_periods = np.insert(sleep_periods, 0, first_period.total_seconds())
    
    if change_points['status'].iloc[-1] == 0:
        last_period = group['timestamp'].iloc[-1] - change_points['timestamp'].iloc[-1]
        sleep_periods = np.append(sleep_periods, last_period.total_seconds())
    
    max_sleep_time = sleep_periods.max() / 60  # 분 단위로 변환
    max_sleep_index = sleep_periods.argmax()
    
    if change_points['status'].iloc[0] == 0:
        start_time = group['timestamp'].iloc[0] if max_sleep_index == 0 else change_points['timestamp'].iloc[max_sleep_index * 2]
    else:
        start_time = change_points['timestamp'].iloc[max_sleep_index * 2 + 1]
    
    end_time = start_time + pd.Timedelta(minutes=max_sleep_time)
    
    return pd.Series({'start_time': start_time, 'end_time': end_time, 'sleep_time': max_sleep_time})

# 그룹별로 수면 시간 계산
result = df.groupby(['algorithm', 'date', 'district_gu', 'district_dong']).apply(calculate_sleep_time).reset_index()

# sleep_time을 "시간:분" 형식으로 변환
result['sleep_time_str'] = result['sleep_time'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}")

# 최종 결과 정리
final_result = result[['algorithm', 'date', 'district_gu', 'district_dong', 'start_time', 'end_time', 'sleep_time_str']]
final_result.columns = ['algorithm', 'date', 'gu', 'dong', 'start_time', 'end_time', 'sleep_time']

# 결과를 CSV로 저장
final_result.to_csv('sleep_time_results_optimized.csv', index=False)
print("완료!")