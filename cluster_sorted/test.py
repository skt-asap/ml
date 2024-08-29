import pandas as pd
import numpy as np
from itertools import combinations

# 데이터 로드
data = pd.read_csv('labeled_data.csv')

print(data['cluster'].unique())