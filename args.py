import argparse

parser = argparse.ArgumentParser(description="Arguments for MNIST")

parser.add_argument('--epoch', help="EPOCH", default=20, type=int)
parser.add_argument('--batch_size', help="BATCH SIZE", default=100, type=int)
parser.add_argument('--weight_init_std', help="WEIGHT INITIALIZATION STD", default=0.01, type=float)
parser.add_argument('--learning_rate', help="LEARNING RATE", default=0.001, type=float)
parser.add_argument('--beta1', help="BETA1", default=0.9, type=float)
parser.add_argument('--beta2', help="BETA2", default=0.999, type=float)