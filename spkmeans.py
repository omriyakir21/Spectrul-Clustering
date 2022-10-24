import math
import sys
from datetime import datetime  # TODO DELETE BEFORE SUBMISSION
import pandas as pd
import numpy as np
import mykmeanssp as km
import enum



def read_data(input):  # reading the data from two files, combine by index and return
    vectors = pd.read_csv(input, dtype=np.float64, header=None)
    vectors = vectors.to_numpy()
    return vectors.tolist()


def kmeans_plus(vectors_mat, k):
    assert 1 < k < len(vectors)  # 1 < k < number(vectors)
    np.random.seed(0)
    N = len(vectors_mat)  # number of vectors
    index = np.random.choice(N)
    centroids = np.zeros((1, len(vectors_mat[index]))) + vectors_mat[index]  # centroids = [ [centroid_1] ]
    list_index = [index]  # collecting the index to return

    for i in range(0, k - 1):  # repeat for k centroids - search new centroid by looking for the most isolated point
        D = np.full((1, N), math.inf)  # D = [[inf], [inf],.... , [inf] ]

        for j in range(0, i + 1):
            tmp = vectors_mat - centroids[j]  # tmp.row[a] = vectors[a] - centroid[j]
            tmp = np.power(tmp, 2)  # tmp[a][b] = ( tmp[a][b]^2 )
            tmp = np.sum(tmp, axis=1)  # tmp[a] = sum(tmp.row[a]) -> tmp[a] = euclid_distance( vector[a], centroid[j] )
            D = np.minimum(D, tmp)  # D[l] = min{ (Xl-hj)^2 | V1<=j<=i ) }

        my_sum = np.sum(D, axis=1)  # calculate sum(D)
        np.true_divide(D, my_sum, out=D)  # for each vector x -> P(x) = D(x)/ sum(D)
        index = np.random.choice(N, p=D[0])  # randomly choosing the next centroid with weights
        list_index.append(index)  # adding new centroid index
        centroids = np.vstack(
            [centroids, vectors_mat[index]])  # adding the new centroid found as a new row to centroids.
    return list_index


def isfloat(num):  # check if input is float
    try:
        float(num)
        return True
    except ValueError:
        return False


# check if the file in correct format
def check_file(input1):
    n1 = len(input1)
    t1 = ((input1[n1 - 4:] == ".txt") or (input1[n1 - 4:] == ".csv"))
    return t1


# print output in the format
def print_all(index_to_print, finalCentroid):
    print(','.join([str(int(index)) for index in index_to_print]))  # printing index of initial centroids
    for centroid in finalCentroid:
        print(','.join(["%.4f" % point for point in centroid]))  # print centroids
    return

#return enum type of the goal from the input
def findGoal(str): 
    if str == "spk":
        return Goal.spk
    elif str == "wam":
        return Goal.wam
    elif str == "ddg":
        return Goal.ddg
    elif str == "lnorm":
        return Goal.lnorm
    elif str == "jacobi":
        return Goal.jacobi
    else:
        return Goal.error

# which goals we have :)
class Goal(enum.IntEnum):
    error = 0
    wam = 1
    ddg = 2
    lnorm = 3
    jacobi = 4
    spk = 5
    clustring = 6

#main function
try:
    
    if len(sys.argv) == 4:
        goal = findGoal(sys.argv[2])
        if sys.argv[1].isnumeric() and int(sys.argv[1]) >= 0 and goal != Goal.error and check_file(sys.argv[3]):
            k = int(sys.argv[1])
            vectors = read_data(sys.argv[3])
            n = len(vectors)
            d = len(vectors[0])
            index_list = [0 for i in range(n)]
            if goal == Goal.spk:
                return_list = km.fit(k, index_list, vectors, n, d, int(goal))
                if return_list is not None:
                    T, k = return_list  # return values from c program
                    T = np.array(T)
                    index_from_func = kmeans_plus(T, k)  # finding initial centroids
                    T = T.tolist()  # converting vectors to list[][] for c module
                    goal = Goal.clustring
                    return_list = km.fit(k, index_from_func, T, n, k,
                                         int(goal))  # running c module and computing centroids
                    if return_list is not None:
                        finalCentroid, k = return_list  # return values from c program
                        print_all(index_from_func, finalCentroid)  # printing results
            else:  # other cases than spk
                km.fit(k, index_list, vectors, n, d, int(goal))
        else:
            print("Invalid Input!", end="")
    else:
        print("Invalid Input!", end="")
except:
    print("An Error Has Occurred", end="")

