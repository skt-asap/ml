import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier

# 데이터 로드
data = pd.read_csv('labeled_data.csv')  # 이전 단계에서 저장한 라벨링된 데이터 파일명

# 범주형 특성 선택
categorical_features = [
    'hour', 'Holiday', 'Equip_800', 'Equip_1800', 'Equip_2100', 
    'Equip_2600_10', 'Equip_2600_20', 'enbid_pci', 'CAnum', 'cluster'
]

# 범주형 변수 인코딩
le = LabelEncoder()
for col in categorical_features:
    data[col] = le.fit_transform(data[col].astype(str))

# 특성과 타겟 분리
X = data[categorical_features]
y = data[['cell_2100', 'cell_2600_10', 'cell_2600_20']]

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성 및 학습
rf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = rf_model.predict(X_test)

# 모델 평가
print("모델 정확도:")
print(accuracy_score(y_test, y_pred))

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['cell_2100', 'cell_2600_10', 'cell_2600_20']))

# 특성 중요도 출력
feature_importance = rf_model.estimators_[0].feature_importances_
for feature, importance in zip(categorical_features, feature_importance):
    print(f"{feature}: {importance}")

# 예시: 새로운 데이터로 예측
new_data = X_test.iloc[0].to_frame().T  # 테스트 데이터의 첫 번째 행을 새로운 데이터로 사용
prediction = rf_model.predict(new_data)
print("\n새로운 데이터에 대한 예측:")
print(prediction)

# 모델 저장
import joblib
joblib.dump(rf_model, 'rf_cell_labeling_model_categorical.joblib')
print("\n모델이 'rf_cell_labeling_model_categorical.joblib' 파일로 저장되었습니다.")