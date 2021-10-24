DEBUG = True

if DEBUG:
    from PIL import Image
    import numpy as np

    def read_iamge(path):
        return np.asarray(Image.open(path).convert('L'))
    
    def write_image(image,path):
        img = Image.fromarray(np.array(image),('L'))
        img.save(path)

from itertools import chain
DIR = "data/"
TEST_BEFORE = 'test/before/'
TEST_AFTER = 'test/after/'

testDataFile = DIR + 't10k-images-idx3-ubyte'  
testLabelsFile = DIR + 't10k-labels-idx1-ubyte'
trainDataFile = DIR + 'train-images-idx3-ubyte' 
trainLabelFile = DIR + 'train-labels-idx1-ubyte' 

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data,'big')

def read_files(filename,max=None):
    images = []
    with open(filename,'rb') as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4)) 
        if max:
            n_images=max
 
        n_rows = bytes_to_int(f.read(4)) 
        n_columns = bytes_to_int(f.read(4))
        for i in range(n_images):
            image = []
            for r in range(n_rows):
                row=[]
                for c in range(n_columns):
                    pixal=f.read(1)
                    row.append(pixal)
                image.append(row)
            images.append(image)
    return images

def read_labels(filename,maxL=None):
    labels = []
    with open(filename,'rb') as f:
        _ = f.read(4)
        n_labels = bytes_to_int(f.read(4)) 
        if maxL:
            n_labels=maxL
        for lblId in range(n_labels):
            label = f.read(1)
            labels.append(label)            
    return labels

def flattenList(l):
    return list(chain.from_iterable(l))
def extractFeatures(X):
    return [flattenList(sample) for sample in X]

def dist(x,y):
    return sum(
        [(bytes_to_int(x_i) - bytes_to_int(y_i))**2 for x_i,y_i in zip(x,y)]
    )**(0.5)

def getDistanceFromSample(X_train,test_sample):
    return [dist(train_sample,test_sample) for train_sample in X_train]

def knn(X_train,y_train,X_test,k=3):
    y_pred = []
    for test_sample_idx,test_sample in enumerate(X_test):
        trainingDist = getDistanceFromSample(X_train,test_sample)
        sortedDistIndices = [
            pair[0]
            for pair in sorted(
                enumerate(trainingDist),
                key = lambda x:x[1]
            )
        ]

        candidates = [
            bytes_to_int(y_train[i])
            for i in sortedDistIndices[:k]
        ]

        #print(f'{test_sample_idx} >> {candidates[0]}')

        
        y_pred.append(candidates[0])
        #here i am returning only the first prediction
        #that is alright if we use large amount of dataset(like 30000+)
        #if we are useing small dataset then it is better to return more candidates and find most occuring one
    return y_pred

def main():
    X_train = read_files(trainDataFile,500)
    y_train = read_labels(trainLabelFile,)
    X_test = read_files(testDataFile,10)
    y_test = read_labels(testLabelsFile)
    
    if DEBUG:
        for idx,test_sample in enumerate(X_test):
            write_image(test_sample,f'{TEST_BEFORE}{idx}before.png')


    X_test = extractFeatures(X_test) 
    X_train = extractFeatures(X_train)  
    
   
    ans=knn(X_train,y_train,X_test)
    listNum=0
    for i in ans:
        print(f'{listNum} >> {i}')
        listNum+=1

    if DEBUG:
        for idx,test_sample in enumerate(X_test):
            write_image(test_sample,f'{TEST_AFTER}{idx}.png')

if __name__=='__main__':
    print('running...')
    main()