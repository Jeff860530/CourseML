import numpy as np
%run magic.ipynb
# or copy paste the cell from https://github.com/tjwei/CrashCourseML/blob/master/DIY_NN/magic.ipynb



# 參考答案
%run solutions/ff_oneline.py


# 請在這裡計算
np.random.seed(1234)


# 參考答案，設定權重
%run -i solutions/ff_init_variables.py
display(A)
display(b)
display(C)
display(d)
display(x)



# 參考答案 定義 relu, sigmoid 及計算 z
%run -i solutions/ff_compute_z.py
display(z_relu)
display(z_sigmoid)




# 參考答案 定義 softmax 及計算 q
%run -i solutions/ff_compute_q.py
display(q_relu)
display(q_sigmoid)




##練習
##設計一個網路:

##輸入是二進位 0 ~ 15
##輸出依照對於 3 的餘數分成三類




# Hint 下面產生數字 i 的 2 進位向量
i = 13
x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
x


# 請在這裡計算
# 參考解答
%run -i solutions/ff_mod3.py


"""
練習
設計一個網路來判斷井字棋是否有連成直線(只需要判斷其中一方即可):
輸入是 9 維向量，0 代表空格，1 代表有下子
輸出是二維(softmax)或一維(sigmoid)皆可，用來代表 True, False
有連線的例子

_X_
X__
XXX

XXX
XX_
_XX

__X
_XX
X__
沒連線的例子

XX_
X__
_XX

_X_
XX_
X_X

__X
_XX
_X_
"""



# 請在這裡計算

#參考答案
%run -i solutions/ff_tic_tac_toe.py

# 測試你的答案
def my_result(x):
    # return 0 means no, 1 means yes
    return (C@relu(A@x+b)+d).argmax()
    # or sigmoid based
    # return (C@relu(A@x+b)+d) > 0

def truth(x):
    x = x.reshape(3,3)
    return (x.all(axis=0).any() or
            x.all(axis=1).any() or
            x.diagonal().all() or
            x[::-1].diagonal().all())

for i in range(512):
    x = np.array([[(i>>j)&1] for j in range(9)])
    assert my_result(x) == truth(x)
print("test passed")