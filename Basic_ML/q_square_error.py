u = train_X[0]#儲存第0個數字
v = train_X[1]#儲存第1個數字

w = train_X[21]#儲存第21個數字
x = train_X[1]#儲存第1個數字

showX(u)
#u的本質是一個很大的矩陣
#print(u)

print("--------------")

showX(v)#v也是個很大的矩陣
#print(v)

print("--------------")

#print(((u - v)**2))
print ( ((u - v)**2).sum() )#直接取方差合
#根據這兩個大矩陣的數字來進行方差若方差很小代表兩個數字的特徵很像

print ( np.linalg.norm(u-v)**2 )#函數取方差合
print("以上不同數字所以方差大")

print("取0為例(取1可能歪歪的方差也很大)")
showX(w)
showX(x)
print ("方差:", ((w - x)**2).sum() )