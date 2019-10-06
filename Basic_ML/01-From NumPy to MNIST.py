from PIL import Image
import numpy as np

import os
import urllib
from urllib.request import urlretrieve
dataset = 'mnist.pkl.gz'
def reporthook(a,b,c):#下面用來顯示下載進度的東西
    print("\rdownloading: %5.1f%%"%(a*b*100.0/c), end="")
    
if not os.path.isfile(dataset):
        origin = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
        print('Downloading data from %s' % origin)
        urlretrieve(origin, dataset, reporthook=reporthook)
        

# 下載完了
import gzip #用來處理壓縮檔
import pickle #
with gzip.open(dataset, 'rb') as f:#gzip.open來打開壓縮檔
    train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
#run -i q_see_mnist_data.py 這行是另外的.py，跑在console，可以看到裡面的[0]是圖片用浮點數形式，[1]是他的答案
# 確認完資料
    
#############
train_set[0].shape

train_set

train_set[0].dtype

train_set[0].min()

train_set[0].max()

train_set[0].mean()
    
import scipy.stats
##################

%run -i q_see_mnist_data.py


scipy.stats.describe(train_set[0])
####


train_X, train_y = train_set
test_X, test_y = test_set


train_y[:20]#就是剛剛的train_set[1]的部分

from IPython.display import display
def showX(X):
    int_X = (X*255).clip(0,255).astype('uint8')  #*255也不大懂，問看看why要做兩次重複的事情，是為了有號無號嗎?
    # N*784 -> N*28*28 -> 28*N*28 -> 28 * 28N
    int_X_reshape = int_X.reshape(-1,28,28).swapaxes(0,1).reshape(28,-1)
    display(Image.fromarray(int_X_reshape))
# 訓練資料， X 的前 20 筆
showX(train_X[:20])

%run -i q_square_error.py 


#########
showX(test_X[0])
tx = test_X[0]

best_i,best_x,best_seq = None,None,784
for x in train_X[100]:
    sqe = np.sum((x-tx)**2)
    if sqe < best_seq:
        best_seq = seq
        best_x = x
        best_i = i
showX(best_x)
print(best_i,best_seq)

((train_X - (x)**2)).sum(axis=1).argmin()

np.argmin(np.sum(np.square(train_X - tx),axis = 1))
##############


%run -i q_find_nn_0.py


%run -i q_find_nn_10.py


%run -i q_small_data.py

((train_X[1] - train_X[0])**2).sum()
########################

np.sum(np.square(train_X[1])- train_X[0])###

np.linalg.norm((train_X[1])- train_X[0])**2

# 資料 normalize
train_X  = train_X / np.linalg.norm(train_X, axis=1, keepdims=True)
test_X  = test_X / np.linalg.norm(test_X, axis=1, keepdims=True)


# 矩陣乘法 == 大量計算內積
A = test_X @ train_X.T
print(A.shape)


A.argmax(axis=1)

A.argmax(axis=1).shape#############

train_y[44566]


predict_y = train_y[A.argmax(axis=1)]

##predict_y = train_y[:20]##############

# 測試資料， X 的前 20 筆
showX(test_set[0][:20])


predict_y[:20]

#測試資料的 y 前 20 筆
test_y[:20]

print("正確率:{}%".format((100*(predict_y[:20] == test_y[:20]).mean())))
# 正確率
#print((predict_y[:20] == test_y[:20]).mean())


from sklearn.decomposition import PCA
pca = PCA(n_components=60)
train_X = pca.fit_transform(train_set[0])
test_X = pca.transform(test_set[0])



train_X.shape


train_X /= np.linalg.norm(train_X, axis=1, keepdims=True)
test_X /= np.linalg.norm(test_X, axis=1, keepdims=True)



# 內積
A = test_X @ train_X.T
predict_y = train_y[A.argmax(axis=1)]
# 正確率
print("正確率:{}%".format((100*(predict_y[:20] == test_y[:20]).mean())))
