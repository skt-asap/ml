import pandas as pd

# 1. 데이터 로드
# 클러스터링 결과 데이터 로드
cluster_data = pd.read_csv('cluster_results_with_location.csv')

# 기존 데이터 로드
original_data = pd.read_csv('ELG_Busan_PoC_per_CA_site_0226_0519.csv')

# 2. 데이터 병합
# 공통 열('cell_id')을 기준으로 데이터 병합
merged_data = pd.merge(original_data, cluster_data[['enbid_pci', 'cluster']], on='enbid_pci', how='left')

# 병합 결과 확인
print("병합된 데이터 프레임의 첫 5개 행:\n", merged_data.head())

# 3. 클러스터별 오름차순 정렬
# 'cluster_label' 열을 기준으로 오름차순 정렬
sorted_data = merged_data.sort_values(by='cluster')

# 정렬된 데이터 확인
print("정렬된 데이터 프레임의 첫 5개 행:\n", sorted_data.head())

# 4. 정렬된 데이터 저장
# 결과를 CSV 파일로 저장
sorted_data.to_csv('sorted_data_with_clusters.csv', index=False)

print("정렬된 데이터가 'sorted_data_with_clusters.csv' 파일로 저장되었습니다.")
