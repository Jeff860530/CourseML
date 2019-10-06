
# 參考範例 softmax regression
A = np.random.normal(size=(20, 784))
B = np.random.normal(size=(20, 1))
E = np.random.normal(size=(10, 20))
F = np.random.normal(size=(10, 1))
n_data = train_X.shape[0]
# 紀錄 loss
loss_history = []
accuracy_history = []
best_accuracy = 0
for epoch in range(5000):    
    idx = np.random.choice(n_data, 300, replace=False)
    X = train_X[idx]
    y = train_y[idx]
    one_y = np.eye(10)[y][..., None]
    C = A @ X + B ##(20*1)
    D = E @ C + F ##(10*1)
    G =  np.exp(D)
    #C = np.exp(A @ X + B)
    #d = W @ X + b
    q = G/G.sum(axis=(1,2), keepdims=True) 
    #loss = -np.log(q[range(len(y)), y]).mean()
    ####更改計算loss
    loss = ((q-one_y)**2).mean()
    #print("loss",loss)
    ####
    loss_history.append(loss)
    accuracy = (q.argmax(axis=1).ravel() == y).mean()
    accuracy_history.append(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    if epoch%100 == 0:
        print(epoch,best_accuracy, loss)
        #print(epoch, accuracy, loss)
    #grad_b_all = q - one_y
    grad_F_all = (2*(q - one_y)**2)
    #print("grad_b_all",grad_b_all)
    grad_F = grad_F_all.mean(axis=0)
    
    grad_C = E.swapaxes(1,2) @ grad_F
        
    grad_E = grad_F @ grad_C.T
    
    grad_B = grad_C
    
    grad_A = grad_C @ X.T
    F -=  grad_F
    C -= grad_C
    E -= grad_E
    B -= grad_B
    A -= grad_A
    