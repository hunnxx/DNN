from model import model
from dataset.mnist import load_mnist

import numpy as np

if __name__ == '__main__':
    model = model()
    _ , testset = load_mnist(normalize=True, one_hot_label=True)
    model.test(testset)