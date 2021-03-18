from args import parser

import numpy as np

args = parser.parse_args()

class Adam:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/adam.py
    # https://twinw.tistory.com/247
    # https://github.com/WegraLee/deep-learning-from-scratch/blob/master/common/optimizer.py
    def __init__(self):
        self.iter = 0
        self.m = None
        self.v = None
    

    def update(self, params, grads):
        if self.m is None:
            self.m = dict()
            self.v = dict()
            for key in grads:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.iter += 1
        lr_t = args.learning_rate * np.sqrt(1.0 - args.beta2**self.iter) / (1.0 - args.beta1**self.iter)

        for key in grads:
            # self.m[key] = args.beta1 * self.m[key] + (1.0 - args.beta1) * grads[key]
            # self.v[key] = args.beta2 * self.v[key] + (1.0 - args.beta2) * grads[key]**2
            # params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            self.m[key] += (1 - args.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - args.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

class GradientDescent:
    def __init__(self):
        pass

    
    def update(self, params, grads):
        for key in grads:
            params[key] -= args.learning_rate * grads[key]