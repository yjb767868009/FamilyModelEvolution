import matplotlib.pyplot as plt
import numpy as np
import os

list_score = []
num_score = 0

with open('log.txt', 'r') as f:
    for line in f.readlines():
        score = float(line.strip().split('    ')[1])
        if score > 0.2:
            num_score += 1
            list_score.append(score)

x = np.linspace(1, num_score, num_score)
plt.figure()
plt.plot(x, list_score)
plt.xlim((1,num_score))
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.savefig('./test.jpg')
plt.show()

