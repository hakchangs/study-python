import numpy as np
from sklearn.base import BaseEstimator

class MyDummyClassifier(BaseEstimator): #상속구현
    
    # 아무것도 학습하지 않음.
    def fit(self, X, y=None):
        pass
    
    # Sex=1 이면 0 아니면, 1로 예측함
    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if X['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1
        return pred

class MyFakeClassifier(BaseEstimator):
    
    # 아무것도 학습하지 않음.
    def fit(self, X, y=None):
        pass
    
    # 모두 0으로 반환
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
