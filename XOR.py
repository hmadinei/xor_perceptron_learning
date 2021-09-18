import numpy as np
from numpy import random
from random import randint

def binary (x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    
def sigmoid (x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative (x):
     return np.exp(-x) / (1 + np.exp(-x)) ** 2

pattern = [[0,0], [0,1], [1,0], [1,1]]
x = np.asarray(pattern)
y = [0, 1, 1, 0]
# random weight 
rand1 = np.random.randint(0, 10, (1, 6))
rand2 = np.random.randint(-10, 0, (1, 2))
w00 = rand1[0][0]
w01 = rand1[0][1]
w02 = rand1[0][2]
w03 = rand1[0][3]
w04 = rand1[0][4]
w05 = rand1[0][5]
w11 = rand2[0][0]
w12 = rand2[0][1]

error = np.zeros(10)
learning_rate = 0.01
# modify the weights until error == 0
while True:
    for i in range(len(x)):
        x1, x2 =  x[i][0], x[i][1]
        y_true = y[i]
        s1 = w00 + x1 * w02 + x2 * w04
        a1 = sigmoid(s1)
        s2 = w01 + x1 * w03 + x2 * w05
        a2 = sigmoid(s2)
        s3 = a1 * w11 + a2 * w12
        y_act = binary(s3)
        error[i] = y_true - y_act 
        if error[i] != 0 : 
            pass
            # modifying the weights 
            w11 += learning_rate * error[i] *  a1
            w12 += learning_rate * error[i] *  a2           
            w00 += learning_rate * (w11 * error[i]) * sigmoid_derivative(s1) * 1
            w01 += learning_rate * (w12 * error[i]) * sigmoid_derivative(s2) * 1
            w02 += learning_rate * (w11 * error[i]) * sigmoid_derivative(s1) * x1
            w03 += learning_rate * (w12 * error[i]) * sigmoid_derivative(s2) * x1
            w04 += learning_rate * (w11 * error[i]) * sigmoid_derivative(s1) * x2
            w05 += learning_rate * (w12 * error[i]) * sigmoid_derivative(s2) * x2
    if np.all((error == 0)):  
        print("w00:", w00)
        print("w01:", w01)
        print("w02:", w02)
        print("w03:", w03)
        print("w04:", w04)
        print("w05:", w05)
        print("-----------------------")
        print("w11:", w11)
        print("w12:", w12)  
        break  
