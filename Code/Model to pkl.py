import time
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

################################################## 피처 불러옴 ##################################################

Train_df= pd.read_csv('Train_StandardScaler_2000000 v2.csv') # 미리 전처리 되어있는 피처 CSV 불러옴 
Train_data = Train_df.iloc[:, 5:] # 사용할 피처 열 
Train_label = Train_df.iloc[:, 0] # Label 열
print(Train_data)
################################################## 머신러닝 실행,학습, pkl 저장 ##################################################
clf = RandomForestClassifier(verbose=2) # 모델 실행
clf.fit(Train_data, Train_label) # 모델 학습
saved_model = pickle.dumps(clf) # 모델을 덤프함
clf_from_pickle = pickle.loads(saved_model) # 덤프함 모델을 pkl 파일로 만듦
joblib.dump(clf_from_pickle, 'RandomForestClassifier_2000000 v2.pkl') 
