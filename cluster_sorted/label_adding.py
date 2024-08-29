import pandas as pd
import numpy as np
from itertools import combinations

# 데이터 로드
data = pd.read_csv('sorted_data_with_clusters.csv')

# 주파수 대역별 RB capacity 설정
RB_capacity = {
    '2100': 75,
    '2600_10': 50,
    '2600_20': 100
}

def label_cell_off(row):
    RB_used = row['RBused']
    RB_total = row['RBtotal']
    
    best_combination = None
    min_RB_total = float('inf')
    
    # 모든 가능한 주파수 대역 조합을 고려
    for r in range(1, len(RB_capacity) + 1):
        for combo in combinations(RB_capacity.keys(), r):
            RB_off = sum(RB_capacity[freq] for freq in combo)
            new_RB_total = RB_total - RB_off
            
            if new_RB_total > 0:
                new_RB_ratio = RB_used / new_RB_total
                if new_RB_ratio <= 0.60 and new_RB_total < min_RB_total:
                    min_RB_total = new_RB_total
                    best_combination = combo
    
    return ','.join(best_combination) if best_combination else 'keep_all'

# 라벨링 수행 및 결과를 새로운 열로 추가
data['label'] = data.apply(label_cell_off, axis=1)

# 라벨링 결과를 개별 열로 확장 (0은 off, 1은 on)
for freq in RB_capacity.keys():
    data[f'cell_{freq}'] = data['label'].apply(lambda x: 0 if freq in x else 1)

# 모든 주파수를 유지하는 경우 확인
data['keep_all'] = (data['label'] == 'keep_all').astype(int)

# 결과 확인 (처음 5개 행만 출력)
print(data[['enbid_pci', 'RBused', 'RBtotal', 'label'] + 
           [f'cell_{freq}' for freq in RB_capacity.keys()] + 
           ['keep_all']].head())

# 각 주파수 대역별로 끌 수 있는 셀의 수 계산
off_counts = {freq: (data[f'cell_{freq}'] == 0).sum() for freq in RB_capacity.keys()}
keep_all_count = data['keep_all'].sum()

print("\n각 주파수 대역별로 끌 수 있는 셀의 수:")
for freq, count in off_counts.items():
    print(f"{freq}: {count} (0은 off, 1은 on)")
print(f"모든 주파수 유지: {keep_all_count}")

# 여러 주파수를 동시에 끄는 경우의 수 계산
multiple_off = data[data['label'].str.contains(',')].shape[0]
print(f"\n두 개 이상의 주파수를 동시에 끄는 셀의 수: {multiple_off}")

# 결과를 CSV 파일로 저장

data.to_csv('labeled_data.csv', index=False)


print("\n주의: 'cell_' 열에서 0은 해당 주파수를 끄는 것을, 1은 켜는 것을 의미합니다.")