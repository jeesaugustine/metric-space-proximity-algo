import os
import numpy as np
from scipy.stats import pareto
from itertools import combinations

def main():
    # file_name = "road-minnesota.mtx"
    # file_name = "road-euroroad.edges"
    file_name = "road-chesapeake.mtx"
    # with open(os.path.join(os.getcwd(), "cpp", "graph_" + str(order_val) + ".txt"), "r") as f:
    with open(os.path.join(os.getcwd(), file_name), "r") as f:
        cpp_graph = f.readlines()
    g = {}
    print("Total Length when recovered: {}".format(len(cpp_graph)))
    ctr = 0
    total_nodes = int(cpp_graph[0])
    del cpp_graph[0]
    one_set = set()

    # One_set node number resetter starting from 0
    node_ids = {}
    i = 0
    for each in cpp_graph:
        l = each.split(' ')
        # u, v, val = int(l[0].strip()), int(l[1].strip()), float(l[-1].strip())
        u, v, val = int(l[0].strip()), int(l[1].strip()), None
        if u not in one_set:
            node_ids[u] = i
            i += 1
            one_set.add(u)
        if v not in one_set:
            node_ids[v] = i
            i += 1
            one_set.add(v)
        if g.get((node_ids[u], node_ids[v]), -1) == -1:
            g[(node_ids[u], node_ids[v])] = val
        else:
            if g[(node_ids[u], node_ids[v])] != val:
                ctr += 1
                print("(u, v): {} -I found this value before: {} and new value is: {} counter: {}".format(
                    (node_ids[u], node_ids[v]),
                    g[(node_ids[u], node_ids[v])],
                    val, ctr))

    # full_mat = graph_maker.out
    alpha = [1]  # list of values of shape parameters
    samples = np.linspace(start=0, stop=5, num=len(g))
    x_m = 1  # scale
    output = None
    for a in alpha:
        output = np.array([pareto.pdf(x=samples, b=a, loc=0, scale=x_m)])
    i = 0
    for each in g:
        if output[0][i] == 0:
            g[each] = 0.002
        else:
            g[each] = (output[0][i])
        i += 1
    c = 0
    for each in output[0]:
        if each >= 0.1:
            c += 1
    print("Total nodes: {} \nTotal Edges: {}\ncounter: {}".format(len(one_set), len(g), ctr))
    print(c)

    filer = open('graph_euro_road.txt', 'w')
    filer.write(str(len(one_set)) + "\n")
    for each in g:
        u, v, val = each[0], each[1], g[each]
        if val >= 1:
            print(u, v, val)
        filer.write(" ".join([str(u), str(v), str(val), "\n"]))
    triangles = list(combinations(list(one_set), 3))
    for triangle in triangles:
        tri = list(combinations(triangle, 2))
        one = tri[0]
        two = tri[1]
        thr = tri[2]
        if g.get((one[0], one[1]), -1) == -1:
            if g.get((one[1], one[0]), -1) == -1:
                continue
            else:
                one = g[(one[1], one[0])]
        else:
            one = g[(one[0], one[1])]
        if g.get((two[0], two[1]), -1) == -1:
            if g.get((two[1], two[0]), -1) == -1:
                continue
            else:
                two = g[(two[1], two[0])]
        else:
            two = g[(two[0], two[1])]
        if g.get((thr[0], thr[1]), -1) == -1:
            if g.get((thr[1], thr[0]), -1) == -1:
                continue
            else:
                thr = g[(thr[1], thr[0])]
        else:
            thr = g[(thr[0], thr[1])]
        assert isinstance(one, float)
        assert isinstance(two, float)
        assert isinstance(thr, float)
        assert one + two >= thr
        assert one + thr >= two
        assert two + thr >= one


        print(tri)

if __name__ == "__main__":
    main()
