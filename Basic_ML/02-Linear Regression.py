from PIL import Image
import numpy as np
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('bmh')
matplotlib.rcParams['figure.figsize']=(8,5)


#簡易的 linear regression 實驗


# 產生隨機數據
X = np.random.normal(0, 3, size=(50,1))
Y = X @ [3] + np.random.normal(0, size=50)


# 畫出來看看
plt.plot(X, Y, 'o');

#用numpy的lstsq
a = np.linalg.lstsq(X, Y)[0]
a


# 畫出來
plt.plot(X, Y, 'o')
plt.plot(X, X @ a, 'o');


%run -i q_lstsq.py




#用 sklearn
from sklearn import linear_model
X = np.random.normal(0, 3, size=(50,1))
Y = X @ [3] + 4 +np.random.normal(0, size=50)
regr = linear_model.LinearRegression()
regr


regr.fit(X,Y)
print(regr.coef_, regr.intercept_)


# 畫出來
plt.plot(X, Y, 'o')
plt.plot(X, regr.predict(X), 'o');
#畫出 test_X = np.linspace(-10,10, 100) 的圖形


%run -i q_linear_test.py


#使用 sklearn 的 datasets
from sklearn import datasets
diabetes = datasets.load_diabetes()
diabetes

import scipy.stats
scipy.stats.describe(diabetes.target)

idx = np.arange(diabetes.data.shape[0])
np.random.shuffle(idx)
X = diabetes.data[idx]
y = diabetes.target[idx]
#試試看 linear regression

train_X = X[:-50, 2:3]
train_y = y[:-50]
test_X = X[-50:, 2:3]
test_y = y[-50:]
regr = linear_model.LinearRegression()
regr.fit(train_X, train_y)
plt.plot(train_X, train_y, 'o');
plt.plot(train_X, regr.predict(train_X), 'o');
np.mean((regr.predict(train_X)-train_y)**2)



plt.plot(test_X, test_y, 'o');
plt.plot(test_X, regr.predict(test_X), 'o');



#用所有變數
train_X = X[:-50]
train_y = y[:-50]
test_X = X[-50:]
test_y = y[-50:]
regr = linear_model.LinearRegression()
regr.fit(train_X, train_y)
np.mean((regr.predict(train_X)-train_y)**2)



np.mean((regr.predict(test_X)-test_y)**2)



plt.plot(test_X[:, 2:3], test_y, 'o');
plt.plot(test_X[:, 2:3], regr.predict(test_X), 'o');



plt.scatter(regr.predict(train_X), train_y, c='g', s=3)
plt.scatter(regr.predict(test_X), test_y, c='b')
plt.plot([0,300],[0,300],'r', linewidth=1);



groups = np.arange(30,300,60)
predict_y=regr.predict(train_X)
plt.boxplot([train_y[(predict_y>=i-30)&(predict_y< i+30)] for i in groups], labels=groups);
plt.plot(np.arange(1,len(groups)+1), groups,'x');


regr = linear_model.Lasso(alpha=0.001)
regr.fit(train_X, train_y)
np.mean((regr.predict(train_X)-train_y)**2)

np.mean((regr.predict(test_X)-test_y)**2)


from sklearn import model_selection
α_space = np.logspace(-4, 0, 50)
scores =[]
for α in α_space:    
    regr.alpha = α
    s = model_selection.cross_val_score(regr, train_X, train_y, cv=3)
    scores.append((s.mean(), s.std()))
scores=np.array(scores).T
plt.semilogx(α_space, scores[0], 'r')
plt.semilogx(α_space, scores[0]+scores[1],'b--')
plt.semilogx(α_space, scores[0]-scores[1],'b--')
plt.fill_between(α_space, scores[0] + scores[1], scores[0] - scores[1], alpha=0.2);


regr = linear_model.LassoCV(alphas = α_space, cv=5)
regr.fit(train_X, train_y)
print(regr.alpha_)
np.mean((regr.predict(train_X)-train_y)**2)


np.mean((regr.predict(test_X)-test_y)**2)

#用 Linear regression 來 classification ?
X = np.random.normal(1, size=(100,1))
y = (X[:,0]>0).ravel()*2-1
regr = linear_model.LinearRegression().fit(X, y)
test_X=np.linspace(-3,3,10).reshape(-1,1)
plt.plot(X, y, 'x');
plt.plot(test_X, regr.predict(test_X), 'r')
plt.plot([-regr.intercept_/regr.coef_[0]]*2, [-1.5,1.5], 'r--')
regr.intercept_


regr.intercept_



#MNIST
import gzip
import pickle
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
    
train_X, train_y = train_set
test_X, test_y = test_set


regr.fit(train_X, train_y)
regr.predict(test_X)


predict_y = np.floor(regr.predict(train_X)+0.5).astype('int').clip(0,9)
np.mean(predict_y == train_y)


predict_y = np.floor(regr.predict(test_X)+0.5).astype('int').clip(0,9)
np.mean(predict_y == test_y)
#準確率約 23% 很低

#One hot encoding
train_Y = np.zeros(shape=(train_y.shape[0], 10))
train_Y[np.arange(train_y.shape[0]), train_y] = 1

train_y[0]


train_Y[0]

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(train_y.reshape(-1,1))
onehot_encoder.transform(train_y.reshape(-1,1)).toarray()[0]


# 訓練模型
regr.fit(train_X, train_Y)

# 用 argmax 得到結果
predict_y = np.argmax(regr.predict(train_X), axis=1)
# 計算正確率
np.mean(predict_y == train_y)

#Q
#試試看 test accuracy
%run -i q_minst_linear_regression.py


#Q
#用 PCA 先處理過 試試看

