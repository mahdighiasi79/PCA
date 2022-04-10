import pandas as pd
import numpy as np
from random import random
import matplotlib.pyplot as plt


class Km:

    def __init__(self, address, k, n):
        self.X = pd.read_csv(address)
        self.K = k
        self.n = n
        self.centers = [[0.0 for i in range(2)] for j in range(self.K)]
        self.idx = []

    def initialize_centers(self):
        sign = 1
        for i in range(self.K):
            self.centers[i][0] = 2 * sign * random()
            sign *= -1
        for i in range(self.K):
            self.centers[i][1] = 2 * sign * random()
            sign *= -1
        return

    def find_closest_centers(self):
        for i in range(len(self.X.index)):
            data = self.X.iloc[i]
            data = np.array(data)
            result = 0
            min_distance = np.inf
            for j in range(self.K):
                center = self.centers[j]
                center = np.array(center)
                distance = np.linalg.norm(center - data, 2)
                if distance < min_distance:
                    min_distance = distance
                    result = j
            self.idx.append(result)
        return

    def compute_means(self):
        count = [0 for i in range(self.K)]
        self.centers = [[0.0 for i in range(2)] for j in range(self.K)]
        for i in range(len(self.X.index)):
            data = self.X.iloc[i]
            count[self.idx[i]] += 1
            self.centers[self.idx[i]][0] += data[0]
            self.centers[self.idx[i]][1] += data[1]
        for i in range(self.K):
            self.centers[i][0] /= count[i]
            self.centers[i][1] /= count[i]
        return

    def execute(self):
        self.initialize_centers()
        for i in range(self.n):
            self.find_closest_centers()
            self.compute_means()
        return

    def sse(self):
        result = 0
        for i in range(len(self.X.index)):
            data = self.X.iloc[i]
            data = np.array(data)
            center = self.centers[self.idx[i]]
            center = np.array(center)
            dist = np.linalg.norm(center - data, 2)
            result += pow(dist, 2)
        return result


if __name__ == "__main__":
    k_means = Km("D:\\computer\\DataMining\\HW4\\Dataset1.csv", 4, 15)

    sse = np.inf
    centers = [[]]
    runs = 50
    for i in range(runs):
        k_means.execute()
        error = k_means.sse()
        if error < sse:
            sse = error
            centers = k_means.centers
    k_means.centers = centers
    k_means.find_closest_centers()

    print(k_means.sse())
    x = k_means.X.X
    y = k_means.X.Y
    x1_line = []
    y1_line = []
    x2_line = []
    y2_line = []
    x3_line = []
    y3_line = []
    x4_line = []
    y4_line = []
    for i in range(len(x)):
        if k_means.idx[i] == 0:
            x1_line.append(x[i])
            y1_line.append(y[i])
        elif k_means.idx[i] == 1:
            x2_line.append(x[i])
            y2_line.append(y[i])
        elif k_means.idx[i] == 2:
            x3_line.append(x[i])
            y3_line.append(y[i])
        else:
            x4_line.append(x[i])
            y4_line.append(y[i])
    plt.plot(x1_line, y1_line, color='blue', marker='o', linestyle='None')
    plt.plot(x2_line, y2_line, color='red', marker='o', linestyle='None')
    plt.plot(x3_line, y3_line, color='green', marker='o', linestyle='None')
    plt.plot(x4_line, y4_line, color='black', marker='o', linestyle='None')
    plt.show()
