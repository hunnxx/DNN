import numpy as np
import cv2 as cv
import sys

from net import net
from args import parser

args = parser.parse_args()

class model:
    def __init__(self):
        self.net = net()


    def train(self, trainset, testset):
        x_train, y_train = trainset

        iteration = int(x_train.shape[0] / args.batch_size)

        for e in range(args.epoch):
            total_loss = 0
            for i in range(iteration):
                # Shuffle 추가
                batch_mask = np.random.choice(x_train.shape[0], args.batch_size)
                x_batch = x_train[batch_mask]
                y_batch = y_train[batch_mask]
                # batch_x = x_train[i*args.batch_size:(i+1)*args.batch_size]
                # batch_y = y_train[i*args.batch_size:(i+1)*args.batch_size]

                _, loss = self.net.forward(x_batch, y_batch)
                self.net.backward()

                total_loss += loss
                print("[{0}][{1}/{2}]".format(e, i, iteration), end='\r', flush=True)
            # print(f"[{e}] : ACC@{self.test(testset)} LOSS@{total_loss / args.batch_size}")
            print("[{0}] : ACC@{1} LOSS@{2}".format(e, self.test(testset), total_loss / args.batch_size))

    def test(self, testset):
        x_test, y_test = testset
        
        output, _ = self.net.forward(x_test, y_test)
        acc = self.net.accuracy(output, y_test)

        return acc