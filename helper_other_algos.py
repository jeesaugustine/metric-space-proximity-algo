import numpy as np
from vanila_algorithms.pam import PAM
from vanila_algorithms.clarans import CLARANS
from vanila_algorithms.prims import Prims
from vanila_algorithms.kruskals import Kruskals

A_path = "graphs/generated/A.txt"
def helper_verify_pam():
    global A_path
    A = np.genfromtxt(A_path)
    B = []
    for i in range(A.shape[0]):
        B.append(np.linalg.norm(A[i, :] - A, axis=1))
    B = np.array(B)
    oracle = lambda x, y: B[x, y]
    n, k, centroids = 1000, 5, [0, 1, 2, 3, 4]
    p = PAM(oracle, n, k, centroids)
    assert len(set(p.centroids).difference(set([877, 844, 253, 446, 336]))) == 0

def helper_verify_clarans():
    global A_path
    A = np.genfromtxt(A_path)
    B = []
    for i in range(A.shape[0]):
        B.append(np.linalg.norm(A[i, :] - A, axis=1))
    B = np.array(B)
    oracle = lambda x, y: B[x, y]
    n, k = 1000, 5
    c = CLARANS(oracle, n, k)
    print(c.centroids)


def helper_verify_prims():
    global A_path
    A = np.genfromtxt(A_path)
    B = []
    for i in range(A.shape[0]):
        B.append(np.linalg.norm(A[i, :] - A, axis=1))
    B = np.array(B)
    oracle = lambda x, y: B[x, y]
    n = 1000
    p = Prims(n, oracle)
    p.mst(0)

def helper_verify_kruskals():
    global A_path
    A = np.genfromtxt(A_path)
    B = []
    for i in range(A.shape[0]):
        B.append(np.linalg.norm(A[i, :] - A, axis=1))
    B = np.array(B)
    oracle = lambda x, y: B[x, y]
    n = 1000
    k = Kruskals(n, oracle)
    k.mst()