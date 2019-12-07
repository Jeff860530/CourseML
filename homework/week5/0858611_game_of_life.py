#########
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

#iniliat the first matrix 
def creat_matrix(sizes):
    matrix = np.random.randint(0,2, size=(sizes,sizes))
    return matrix

'''
def Padding(matrix):
    padding_matrix = np.pad(matrix, 1, 'constant', constant_values = 0)
    return  padding_matrix
'''
#padding the another side number 
def Padding(matrix):
    sizes = matrix.shape[0]
    new_padding_matrix = np.random.randint(0,1, size=(sizes*3,sizes*3))
    padding_matrix = np.pad(matrix, sizes, 'constant', constant_values = 0)
    new_padding_matrix = new_padding_matrix + padding_matrix
    new_padding_matrix = new_padding_matrix + np.roll(padding_matrix, sizes, axis=0) + np.roll(padding_matrix, -1*sizes, axis=0)
    new_padding_matrix = new_padding_matrix + np.roll(new_padding_matrix, sizes, axis=1) + np.roll(new_padding_matrix,-1*sizes, axis=1)
    padding_matrix = new_padding_matrix[sizes-1:sizes*2+1,sizes-1:sizes*2+1]
    return padding_matrix


#def the game
def Game_of_live(matrix):
    sizes = matrix.shape[0]
    result_matrix = np.random.randint(0,1, size=(sizes,sizes))
    padding_matrix = Padding(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cross_matrix = padding_matrix[i:i+3, j:j+3] * test_matrix
            if (cross_matrix.sum() < 2):
                result_matrix[i,j] = 0
            elif (cross_matrix.sum() == 2 or cross_matrix.sum() == 3):
                result_matrix[i,j] = 1
            elif (cross_matrix.sum() > 3):
                result_matrix[i,j] = 0
    rtn = result_matrix
    return rtn


########
if __name__ == '__main__': 
    test_matrix = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])
    matrix = creat_matrix(20) #input the size of square matrix
    for i in range (30):     #input the times of Game_of_live
        rnt = Game_of_live(matrix)
        matrix = rnt
        plt.imshow(matrix)
        plt.show()

##########
# 下面來定義 CNN 網路
from keras.models import Model
from keras.layers import Conv2D, Input

# 權重
def W(size):    
    rtn = np.ones(shape=(3,3,1,4))
    rtn[1,1,0,2:] = 9
    
    return rtn

def b(size):
    #return Random_Matrix(1,4,-12,0).reshape(4)   
    print(size)
    return np.array([-2,-3, -11,-12])

def W2(size):
    return np.array([1,-2,1,-2]).reshape(1,1,4,1)

def b2(size):
    # just to be safe
    #bias2 = np.random.randn()
    #return np.full(size,bias2)
    print(size)
    return np.full(size, -0.5)
    

# 網路模型定義
inputs = Input(shape=(None,None,1))
hidden = Conv2D(filters=4, kernel_size=3, padding='same', activation="relu",
             kernel_initializer=W, bias_initializer=b)(inputs)
out = Conv2D(filters=1, kernel_size=1, padding='same', activation="relu",
             kernel_initializer=W2, bias_initializer=b2)(hidden)
model = Model(inputs, out)

bias = model.get_weights()[1]
print(bias)
bias2 = model.get_weights()[3]
print(bias2)

######
# 檢查看看結果是否正確
N = 100
# 隨機 100x100 盤面
boards = np.random.randint(0,2, size=(N,5,5))
# 用 CNN 模型跑下個盤面
rtn = model.predict(boards[..., None])
# >0 的值當成活著， <0 的值當成死的 (應該不會有 0的值)
rtn = (rtn>0).astype('int')
# 一一檢查
for i in range(N):
    b = Game_of_live(boards[i])
    print(b)
    print(rtn[i, :, :, 0])
    print("-----------")
    assert (b == rtn[i, :, :, 0]).all()
    print("OK", i)
    
    
    

