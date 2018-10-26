import numpy as np

labels = np.loadtxt('test_labels.txt', dtype = np.int)
gt = np.zeros((1,303))
for i, row in enumerate(labels):
    for j,label in enumerate(row):
        if label == 1:
            gt[0][j]+=1

print(gt[0])
f = open('ground_truth.txt', 'w')
for g in gt[0]:
    f.write('%d'%g)
    f.write(' ')
f.close()
