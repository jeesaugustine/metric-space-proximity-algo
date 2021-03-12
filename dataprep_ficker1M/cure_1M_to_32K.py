import os
from skimage import io
import numpy as np
import math


def get_image_vector(name):
    return io.imread(os.path.join("32k/", name), as_gray=True).flatten()


def get_min_rank():
    par = os.listdir("32k/")
    par = list(filter(lambda x: ".jpg" in x, par))
    par.sort()
    mi = math.inf
    for img in par:
        if ".jpg" in img:
            mi = min(mi, get_image_vector(img).shape[0])
    return mi


def get_oracle():
    par = os.listdir("32k/")
    par = list(filter(lambda x: ".jpg" in x, par))
    par.sort()

    img_vector = []
    for p in par:
        img_vector.append(get_image_vector(p))
    # img_vector = list(map(lambda x: x[:4096], img_vector))
    # Change the above line for higher dimensions and comment the lines below
    img_vector = list(map(lambda x: x[:2], img_vector))
    a = np.array(img_vector)
    # a = a.astype(np.float64)

    def oracle(u, v):
        # val = np.linalg.norm(a[u, :4096]-a[v, :4096], 2)/15519
        # print("u: {}, v: {}: {}".format(u, v, val))
        # return np.linalg.norm(a[u, :4096]-a[v, :4096], 2)/64
        return np.linalg.norm(a[u, :2] - a[v, :2], 2) / 64
        # return val
    return oracle


def get_normalized():
    par = os.listdir("32k/")
    par = list(filter(lambda x: ".jpg" in x, par))
    maximum = 0
    oracle = get_oracle()
    for i in range(len(par)):
        print(i)
        for j in range(i+1, len(par)):
            maximum = max(oracle(i, j), maximum)
    return maximum


if __name__ == "__main__":
    # print("Min val: {}".format(get_min_rank()))
    # print("Max val: {}".format(get_normalized()))
    pass




