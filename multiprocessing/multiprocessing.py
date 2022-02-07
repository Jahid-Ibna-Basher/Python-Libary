from multiprocessing import Pool


pipline = Pool(processes=4)

counter = 0

def count(c):
    return c+1


for i in range(5):
    print(pipline.map(count(), [1,2,3]))