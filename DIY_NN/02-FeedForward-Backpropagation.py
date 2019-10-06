import numpy as np
%run -i solutions/magic.py

# 參考範例， 各種函數、微分
%run -i solutions/ff_funcs.py

# 參考範例， 計算 loss
%run -i solutions/ff_compute_loss2.py


# 計算 gradient
%run -i solutions/ff_compute_gradient.py


# 更新權重，計算新的 loss
%run -i solutions/ff_update.py

%matplotlib inline
import matplotlib.pyplot as plt

# 參考範例
L_history=[]
%run -i solutions/ff_train_mod3.py
plt.plot(L_history);


# 訓練結果測試
for i in range(16):
    x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
    y = i%3
    U = relu(A@x+b)
    q = softmax(C@U+d)
    print(q.argmax(), y)
    
    
def truth(x):
    x = x.reshape(3,3)
    return int(x.all(axis=0).any() or
            x.all(axis=1).any() or
            x.diagonal().all() or
            x[::-1].diagonal().all())
    
    
%run -i solutions/ff_train_ttt.py
plt.plot(accuracy_history);


