def softmax(x):
    t = np.exp(x)
    return t/t.sum()
import numpy as np
%run magic.ipynb

W = Matrix([-1,-1,0,0], [1,-1,0,0], [-1,1,0,0], [1,1,0,0])
b = Vector(0,0,0,0)
count = 0
for i in range(16):
    x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
    r = W @ x + b
    print("i=", i, "predict:", r.argmax(), "ground truth:", i%4)
    if( r.argmax()== (i%4) ):
      count+=1
print("Accuracy:{}%".format(count*100/16))