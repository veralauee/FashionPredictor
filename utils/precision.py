import numpy as np

gt = np.loadtxt('ground_truth.txt', dtype = np.int)
def calculate(filename, th):
    avg = 0
    cnt = 0
    zero=0
    invalid = []
    valid = []
    f = open(filename).readlines()

    for i,line in enumerate(f):
        line = line.split(': ')
        if len(line)<2:
            continue
        precision = float(line[1])
        if precision <=th:
            invalid.append(i)
            zero +=1
            continue

        cnt +=1
        valid.append(i)
        avg += float(precision)
    avg = float(avg)/cnt
    print avg
    print ('valid: %d' %cnt)
    print valid
    print ('invalid: %d '%zero)

print('------------------ top3_precision -------------------')    
calculate("top3_precision.txt", 0.3)
print ('\n')
print('----------------- top5_precision -----------------')
calculate("fine_tune_top5.txt", 0)

print('----------------- top10_precision ----------------')
calculate("top10_precision.txt", 0.3)
