#line1~30 is Course-practice
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
print("以上存取數字5和0不同數字所以方差大")

print("取兩個0為例(取1可能歪歪的方差也很大)")
showX(w)
showX(x)
print ("方差:", ((w - x)**2).sum() )
print("以上了解找nearest neighbor的方法==>使用方差合最小來找最小特徵(矩陣的數字)差異最小者")

#the following is homework1
print("-----------------------------------------------------------------")
print("Question:Use nearest neighbor method to do handwritten digit recognition ")
print("顯示要找的圖片test_X[0]的圖案")
showX(test_X[0])
#showX(train_X[10000000])會顯示error
#train內只有50000個數據

print("計算方差")
print("找出train_X中每個數據 與 test_X[0]的方差合")
Variance = ((train_X - test_X[0])**2).sum(axis=1)

print("找出方差合最小的 index (訓練集合中最像test_X[0]的項目)")
idx = Variance.argmin()

print("找到的train集合中 第38620個數據")
print("train_X[{}]".format(idx))

print("show出訓練集中最像的圖片")
showX(train_X[idx])

print("印出訓練集合第38620個數據 和 標籤內容(train_y[index])")
print("train_X[{}] = {}".format(idx, train_y[idx]))

print("印出標籤")
print("test_y[0] =", test_y[0])
