# 顯示 test_X[0]的圖案
showX(test_X[0])
# 計算方差
#showX(train_X[10000000])會顯示error
#train內只有50000個數據

#找出train_X中每個數據和test_X[0]的方差合
Variance = ((train_X - test_X[0])**2).sum(axis=1)

# 找出方差最小的 index (訓練集合中最像test的項目)
idx = Variance.argmin()

#輸出找到的第idx個數據
print("train_X[{}]".format(idx))

#show圖案
showX(train_X[idx])

#印出第idx個數據 和 標籤內容再trainData的第[idx]的y
print("train_X[{}] = {}".format(idx, train_y[idx]))

#印出test的第index 0項 
print("test_y[0] =", test_y[0])
