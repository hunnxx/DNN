import numpy as np
import cv2 as cv

from optimizer import Adam
from args import parser

args = parser.parse_args()

class net:
    def __init__(self):
        self.params = dict()
        self.grads = dict()

        self.params['W1'] = args.weight_init_std * np.random.randn(784, 256)
        self.params['B1'] = np.zeros(256)
        self.params['W2'] = args.weight_init_std * np.random.randn(256, 256)
        self.params['B2'] = np.zeros(256)
        self.params['W3'] = args.weight_init_std * np.random.randn(256, 10)
        self.params['B3'] = np.zeros(10)

        self.optimizer = Adam()


    def forward(self, input, target):
        z1 = np.dot(input, self.params['W1']) + self.params['B1']
        x2 = self.relu(z1)
        z2 = np.dot(x2, self.params['W2']) + self.params['B2']
        x3 = self.relu(z2)
        z3 = np.dot(x3, self.params['W3']) + self.params['B3']
        output = self.softmax(z3)

        self.params['X1'] = input
        self.params['Z1'] = z1
        self.params['X2'] = x2
        self.params['Z2'] = z2
        self.params['X3'] = x3
        self.params['Z3'] = z3
        self.params['Y'] = output
        self.params['T'] = target

        return output, self.loss(output, target)


    def backward(self):
        # Grad
        # https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy

        W3 = (self.params['Y'] - self.params['T']) / args.batch_size
        self.grads['W3'] = np.dot(self.params['X3'].T, W3)
        self.grads['B3'] = np.sum(W3, axis=0)

        W2 = np.dot(W3, self.params['W3'].T)
        W2 *= np.where(self.params['Z2'] > 0, 1, 0)
        self.grads['W2'] = np.dot(self.params['X2'].T, W2)
        self.grads['B2'] = np.sum(W2, axis=0)
        
        W1 = np.dot(W2, self.params['W2'].T)
        W1 *= np.where(self.params['Z1'] > 0, 1, 0)
        self.grads['W1'] = np.dot(self.params['X1'].T, W1)
        self.grads['B1'] = np.sum(W1, axis=0)

        self.optimizer.update(self.params, self.grads)
        

    def loss(self, output, target):
        # Cross Entropy
        if output.ndim == 1:
            target = t.reshape(1, target.size)
            output = output.reshape(1, output.size)
        
        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if target.size == output.size:
            target = target.argmax(axis=1)
             
        batch_size = output.shape[0]
        return -np.sum(np.log(output[np.arange(batch_size), target] + 1e-7)) / batch_size


    def accuracy(self, output, target):
        return np.sum(np.argmax(output, axis=1) == np.argmax(target, axis=1)) / output.shape[0]


    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x) # 오버플로 대책
        return np.exp(x) / np.sum(np.exp(x))


    def relu(self, x):
        return np.maximum(0, x)


    def relu6(self, x):
        return np.minimum(6, np.maximum(0, x))