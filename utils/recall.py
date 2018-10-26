import numpy as np

def calculate(filename):
    avg = 0
    cnt = 0

    invalid = []
    f = open(filename).read().splitlines()

    for i,line in enumerate(f):
        recall = float(line)
        if recall <=0.4:
            invalid.append(i)
            continue

        cnt +=1
        avg += float(recall)
    avg = float(avg)/cnt
    print avg
    print ('valid: %d' %cnt)
    print ('invalid: %d '% len(invalid))
    #print(invalid)

print('----------------- top3 recall rate -------------------')
calculate("top3_recall.txt")
print('\n')
print('------------------ top5 recall rate-----------------')
calculate("top5_recall.txt")
