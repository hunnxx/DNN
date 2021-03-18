from model import model
from dataset.mnist import load_mnist

if __name__ == '__main__':
    model = model()
    trainset, testset = load_mnist(normalize=True, one_hot_label=True)
    model.train(trainset, testset)