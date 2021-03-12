import math
import os
import sys
import time

from Landmark_Edge import LandMark
from bottom_up_tree_sample_min_max import BottomUPTree
from graph_maker import NlogNGraphMaker
from helper import _get_matrix_from_adj_dict
from lower_bound_tree import LBTree
from sasha_wang import *
from unified_graph_lb_ub import unified_graph_lb_ub

sys.setrecursionlimit(20000)

order_val = int(sys.argv[1])
print(order_val)

# graph_maker = NlogNGraphMaker(order_val)
# g = graph_maker.get_nlogn_edges()
with open(os.path.join(os.getcwd(), "cpp", "graph_" + str(order_val) + ".txt"), "r") as f:
    cpp_graph = f.readlines()
g = {}
print("Total Length when recovered: {}".format(len(cpp_graph)))
ctr = 0
one_set = set()
for each in cpp_graph:
    l = each.split(' ')
    u, v, val = int(l[0].strip()), int(l[1].strip()), float(l[-1].strip())
    one_set.add(u)
    one_set.add(v)
    if g.get((u, v), -1) == -1:
        g[(u, v)] = val
    else:
        if g[(u, v)] != val:
            ctr += 1
            print("(u, v): {} -I found this value before: {} and new value is: {} counter: {}".format((u, v), g[(u, v)], val, ctr))
# full_mat = graph_maker.out
print("Total nodes: {} \nTotal Edges: {}\ncounter: {}".format(len(one_set), len(g), ctr))
_nodes = set()
for (u, v) in g.keys():
    _nodes.add(u)
    _nodes.add(v)
assert len(_nodes) == order_val
print("Validation Complete...")
g_mat = _get_matrix_from_adj_dict(g, order_val)

# Check what the function does at the bottom until commented
# start = time.time()
# k = math.ceil(math.log(order_val, 1.8))
# le = LandMark(g, order_val, k)
# print("K: ", k)
# le.get_edge_landmark()
# le.landmark_dijk()
# lb_mat, ub_mat = le.mat_look_up()
# end_time = time.time()
# print("Landmarks : ", end_time-start)


start = time.time()
bu_tree = BottomUPTree(g, order_val)
bu_tree.build_tree()
tree = bu_tree.tree
lbt = LBTree(tree, g_mat)
end_time = time.time()
print("Time - LBT : ", end_time-start)

lb = 0
ub = 0
lb_lbt = 0
lb_lbub = 0
lb_lbub1 = 0

# start = time.time()
# obj = SashaWang()
# obj.store(g, order_val)
# end_time = time.time()
# print("Time - SW : ", end_time-start)

start = time.time()
obj1 = unified_graph_lb_ub()
obj1.store(g, order_val)
end_time = time.time()
print("Time - LBT with Upper bound : ", end_time-start)

start = time.time()
obj2 = unified_graph_lb_ub(random_tree=True)
obj2.store(g, order_val)
end_time = time.time()
print("Time - LBT with Upper bound (Random): ", end_time-start)

# s_w_lb = 0
graph_weights = 0
for i in range(order_val):
    # print(i)
    for j in range(i + 1, order_val):
        # lb += obj.lb_matrix[i][j] - lb_mat[i][j]
        # ub += ub_mat[i][j] - obj.ub_matrix[i][j]
        # lb_lbt += obj.lb_matrix[i][j] - lbt.lb_matrix[i][j]
        # lb_lbub += obj.lb_matrix[i][j] - obj1.lb_matrix[i][j]
        # lb_lbub1 += obj.lb_matrix[i][j] - obj2.lb_matrix[i][j]
        # lb_lbt += lbt.lb_matrix[i][j]
        # lb_lbub += obj1.lb_matrix[i][j]
        # lb_lbub1 += obj2.lb_matrix[i][j]
        if (g.get((i, j), -1) == -1) and (g.get((j, i), -1) == -1):
            # s_w_lb += obj.lb_matrix[i][j]
            lb_lbt += lbt.lb_matrix[i][j]
            lb_lbub += obj1.lb_matrix[i][j]
            lb_lbub1 += obj2.lb_matrix[i][j]
        else:
            graph_weights += g[(i, j)]
# print("LB Diff: ", lb/(order_val * (order_val - 1) / 2 - len(g)))
# print("UB Diff: ", ub/(order_val * (order_val - 1) / 2 - len(g)))
# print("LB Diff LBT: ", lb_lbt/(order_val * (order_val - 1) / 2 - len(g)))
# print("LB Diff LBUB: ", lb_lbub/(order_val * (order_val - 1) / 2 - len(g)))
# print("LB Diff LBUBRandom: ", lb_lbub1/(order_val * (order_val - 1) / 2 - len(g)))
print("*"*50)
# print("LB Diff LBT: ", lb_lbt)
# print("LB Diff LBUB: ", lb_lbub)
# print("LB Diff LBUBRandom: ", lb_lbub1)
# print("-"*50)
print("LB LBT: ", lb_lbt)
print("LB LBUB: ", lb_lbub)
print("LB LBUB(Random): ", lb_lbub1)
print("Original graph weight is: {}".format(graph_weights))
# print("Original graph weight is: {} & Sasha Wang Sum LB is: {}".format(graph_weights, s_w_lb))
print("-"*50)

# if lb_lbt >= lb and lb_lbub >= lb:
#     print("Landmarks")
# elif lb_lbub >= lb_lbt:
#     print("LBTree")
# else:
#     print("Tree algorithm")
print("=" * 50)