# -*- coding: utf-8 -*

# 对象似然函数
import numpy as np

def likelihood_sub(x, y, beta):
    return -y * np.dot(beta, x.T) + np.math.log(1 + np.math.exp(np.dot(beta, x.T)))   

def likelihood(X, y, beta):
    sum = 0
    m,n = np.shape(X)  
    
    for i in range(m):
        sum += likelihood_sub(X[i], y[i], beta)
                                                 
    return sum

def partial_derivative(X, y, beta):
    m,n = np.shape(X) 
    pd = np.zeros(n)
    
    for i in range(m):
        tmp = y[i] - sigmoid(X[i], beta)
        for j in range(n):
            pd[j] = pd[j] + (tmp) * X[i][j]                                          
    return pd
       
# 基础梯度算法的实现
def gradDscent_1(X, y):
    import matplotlib.pyplot as plt  

    h = 0.1  # 迭代步长
    max_times= 500  # 迭代次数限制 
    m, n = np.shape(X)
    
    b = np.zeros((n, max_times))  # 参数的收敛曲线         
    beta = np.zeros(n)  # 参数与初始值  
    delta_beta = np.ones(n)*h
    llh = 0
    llh_temp = 0
    
    for i in range(max_times):
        beta_temp = beta.copy()
        
        for j in range(n): 
            # 求偏导
            beta[j] += delta_beta[j]
            llh_tmp = likelihood(X, y, beta)
            delta_beta[j] = -h * (llh_tmp - llh) / delta_beta[j]
            
            b[j,i] = beta[j] 
            
            beta[j] = beta_temp[j]
            
        beta += delta_beta            
        llh = likelihood(X, y, beta)

    t = np.arange(max_times)
    
    f2 = plt.figure(3) 
    
    p1 = plt.subplot(311)
    p1.plot(t, b[0])  
    plt.ylabel("w1")  
    
    p2 = plt.subplot(312)
    p2.plot(t, b[1])  
    plt.ylabel("w2")  
        
    p3 = plt.subplot(313)
    p3.plot(t, b[2])  
    plt.ylabel("b")  
        
    plt.show()               
    return beta

#随机梯度算法的实现
def gradDscent_2(X, y):  
    import matplotlib.pyplot as plt  

    m, n = np.shape(X)
    h = 0.5  #  迭代的步长
    beta = np.zeros(n)  # 参数
    delta_beta = np.ones(n) * h
    llh = 0
    llh_temp = 0
    b = np.zeros((n, m))  # 参数的收敛曲线

    for i in range(m):
        beta_temp = beta.copy()
        
        for j in range(n): 
            # 求偏导
            h = 0.5 * 1 / (1 + i + j)  # 改变迭代的步长
            beta[j] += delta_beta[j]
            
            b[j,i] = beta[j]
            
            llh_tmp = likelihood_sub(X[i], y[i], beta)
            delta_beta[j] = -h * (llh_tmp - llh) / delta_beta[j]   
            
            beta[j] = beta_temp[j]  
               
        beta += delta_beta    
        llh = likelihood_sub(X[i], y[i], beta)
              
    t = np.arange(m)
    
    f2 = plt.figure(3) 
    
    p1 = plt.subplot(311)
    p1.plot(t, b[0])  
    plt.ylabel("w1")  
    
    p2 = plt.subplot(312)
    p2.plot(t, b[1])  
    plt.ylabel("w2")  
        
    p3 = plt.subplot(313)
    p3.plot(t, b[2])  
    plt.ylabel("b")  
        
    plt.show()   
            
    return beta

def sigmoid(x, beta):
    return 1.0 / (1 + np.math.exp(- np.dot(beta, x.T)))  
    
def predict(X, beta):
    m, n = np.shape(X)
    y = np.zeros(m)
    
    for i in range(m):
        if sigmoid(X[i], beta) > 0.5: y[i] = 1;      
    return y
                            
    return beta
