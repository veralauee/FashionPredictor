import scipy.io as sio
from scipy.spatial import distance
import numpy as np

# find topk neighbors, return index
def find_neighbor(test, mat, topk):
    alldst = []

    for i in range(len(mat)):
        dst = distance.euclidean(test[0], mat[i][0])
        alldst.append(dst)
        
    neighbors = sorted(range(len(alldst)), key=lambda i: alldst[i], reverse=False)[:topk]
    return neighbors

def resize(mat):
    ans = []
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            one = np.reshape(mat[i][j], (1,8192))
            ans.append(one)
    return np.array(ans)


def main():
    # load mat
    consumer = sio.loadmat('consumer_item.mat')['FC'][0]
    shop = sio.loadmat('shop_item.mat')['FC'][0]
    #consumer = resize(consumer)
    #shop = resize(shop)

    topk = 3

    TP = 0 # TRUE Positive
    print consumer.shape
    print shop.shape
    
    for i,batch in enumerate(consumer):
        one_consumer = np.reshape(batch, (1,8192))
                        
        neighbors = find_neighbor(one_consumer, shop, topk)
        print neighbors
        if i in neighbors:
            TP += 1
            if i%200 ==0:
                print neighbors
                print('retrieved images', TP)
                print('recall rate', float(TP/float(i+1)))
    
    
    recall_rate = float(TP/float(i+1))
    print('retrieved images', TP)
    print('recall rate', recall_rate)
            
if __name__ == '__main__':
    main()
        
