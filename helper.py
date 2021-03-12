from node import Node
from sasha_wang import *
import time
from graph_maker import NlogNGraphMaker
from bottom_up_tree_sample import BottomUPTree
from top_down_tree_sampling import TopDownTree
import copy
from lower_bound_tree import LBTree
import random
import numpy as np
from dijkstra import Dijkstra

debug = False
order_val = 1000

def _get_dbl_level_dict(g_dict, order_val=0):
    dbl_dict = {}
    for k, val in g_dict.items():
        i, j = k
        if i not in dbl_dict:
            dbl_dict[i] = {}
        if j not in dbl_dict:
            dbl_dict[j] = {}
        dbl_dict[i][j] = val
        dbl_dict[j][i] = val
    for i in range(order_val):
        if i not in dbl_dict:
            dbl_dict[i] = {}
    return dbl_dict

def _put(dict, key, value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]

def _make_neighbours(dict):
    neighbours = {}
    for k in dict.keys():
        _put(neighbours, k[0], k[1])
        _put(neighbours, k[1], k[0])
    return neighbours

def _bfs(node, dict, n):
    queue = []
    neighbour_dict = _make_neighbours(dict)
    queue.append(node)
    visited = [False] * n
    visited[node] = True
    total = 1
    while len(queue) > 0:
        head = queue.pop(0)
        neighbours = neighbour_dict[head]
        for i in neighbours:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
                total += 1
    #print(total)
    return total == n

def _get_matrix_from_adj_dict(dict, n):
    mat = [[0]*n for i in range(n)]
    for k in dict.keys():
        x, y = k
        mat[x][y] = mat[y][x] = dict[k]
    return mat

graph_maker = NlogNGraphMaker(order_val)
g = graph_maker.get_nlogn_edges()
full_mat = graph_maker.out
g_mat = _get_matrix_from_adj_dict(g, order_val)
_bfs(0, g, order_val)

def get_nodes(node):
    if not node:
        return []
    nodes = []
    def _get_nodes(n):
        nodes.append(n.index)
        for c in n.children:
            _get_nodes(c)

    _get_nodes(node)
    return nodes

def helper_ub():
    global g, g_mat, order_val, full_mat
    obj = SashaWang()
    obj.store(g, order_val)
    ub_sw = obj.ub_matrix
    ub = copy.copy(g_mat)
    start = time.time()
    dbl_dict = _get_dbl_level_dict(g)
    d = Dijkstra(dbl_dict, order_val)
    nodes = list(range(order_val))
    for i in nodes:
        sp = d.shortest_path(nodes, i)
        for index in range(order_val):
            # distance, node, parent
            ub[i][index] = sp[index][0]
            ub[index][i] = sp[index][0]
    end = time.time()
    ub_sw_np = np.array(ub_sw)
    assert not np.any(np.abs(ub_sw_np - ub) > 0.0001)
    print("Total time Dijkstra", end-start)

def helper_lb_td_tree_verify(levels=1):
    global g, g_mat, order_val, full_mat
    out = full_mat
    size_of_graph = order_val
    lb_mat = np.copy(g_mat)
    lb_mat_orig = np.copy(g_mat)
    start = time.time()
    tree_build_time = 0
    td_tree = TopDownTree(g, size_of_graph)
    cur = 1
    queue = [list(range(order_val))]
    while cur < levels:
        queue_new = []
        for nodes in queue:
            if len(nodes) < 5:
                continue
            start_tree_time = time.time()
            td_tree.make_tree(nodes)
            td_tree.build_tree()
            tree_build_time += time.time() - start_tree_time
            tree = td_tree.tree
            lb = LBTree(tree, lb_mat, False)
            for c in tree.children:
                queue_new.append(get_nodes(c))
        queue = queue_new
        cur += 1
    if debug:
        print(_bfs(td_tree.root, td_tree.tree_dict, size_of_graph))
        print(len(td_tree.tree_dict))
    end = time.time()
    print("Total time : ", end - start)
    print("Tree Building Time: ", tree_build_time)
    x = np.array(lb.lb_matrix)
    y = np.array(lb_mat_orig)
    total_changed = np.sum(np.abs(x - y) >= 0.01)
    print("Total changed", total_changed)
    for i in range(size_of_graph):
        for j in range(size_of_graph):
            lower_bound = lb.lb_matrix[i][j]
            assert lower_bound <= out[i][j]

def helper_lb_bu_tree_verify():
    global g, g_mat, order_val, full_mat
    out = full_mat
    size_of_graph = order_val
    lb_mat = np.copy(g_mat)
    lb_mat_orig = np.copy(g_mat)
    start = time.time()
    bu_tree = BottomUPTree(g, size_of_graph)
    if debug:
        print(_bfs(bu_tree.root, bu_tree.tree_dict, size_of_graph))
        print(len(bu_tree.tree_dict))
    bu_tree.build_tree()
    tree_build_time = time.time()
    print("Tree Building Time: ", tree_build_time-start)
    tree = bu_tree.tree
    lb = LBTree(tree, lb_mat)
    end = time.time()
    print(end - tree_build_time)
    x = np.array(lb.lb_matrix)
    y = np.array(lb_mat_orig)
    total_changed = np.sum(np.abs(x - y) >= 0.01)
    print("Total changed",total_changed)
    for i in range(size_of_graph):
        for j in range(size_of_graph):
            lower_bound = lb.lb_matrix[i][j]
            assert lower_bound <= out[i][j]

def printer_matrix(matrix):
    for i in range(len(matrix)):
        print(' '.join(map(lambda x: "{:.3f}".format(x), matrix[i])))

def helper_verify(obj):
    import random
    global order_val, g, full_mat, debug
    print("SASHA WANG")
    order = order_val
    out = full_mat
    val_dict = g
    start = time.time()
    obj.store(val_dict, order)
    end = time.time()
    print(end - start)
    for i in range(order):
        for j in range(order):
            lb, ub = obj.lookup(i, j)
            assert lb <= out[i][j] <= ub
    for num in range(100):
        i = random.choice(list(obj.uncalculated.keys()))
        j = random.choice(list(obj.uncalculated[i]))
        obj.update([i, j], out[i, j])
        for i in range(order):
            for j in range(order):
                lb, ub = obj.lookup(i, j)
                assert lb <= out[i][j] <= ub
    end = time.time()
    print(end - start)
    if debug:
        print("Actual values matrix")
        printer_matrix(obj.matrix)
        print("UB values matrix")
        printer_matrix(obj.ub_matrix)
        print("LB values matrix")
        printer_matrix(obj.lb_matrix)

def prims(matrix):
    from scipy.sparse.csgraph import minimum_spanning_tree
    Tcsr = minimum_spanning_tree(matrix)
    MST = Tcsr.toarray().astype(type(matrix[0][0]))
    return MST

def helper_lb_verify():
    global order_val, g_mat, full_mat, debug
    size_of_graph = order_val
    out = full_mat
    print("LB TREE")
    MST = prims(g_mat)
    mst_symm = MST + MST.T
    start = time.time()
    root_index = random.choice(list(range(MST.shape[0])))
    tree = make_tree(root_index, mst_symm, set([root_index]))
    mst_copy = np.copy(mst_symm)
    lb = LBTree(tree, mst_copy)
    end = time.time()
    print(end - start)
    if debug:
        print("Actual values matrix")
        printer_matrix(out)
        print("MST values matrix")
        printer_matrix(mst_symm)
        print("LB values matrix")
        printer_matrix(lb.lb_matrix)
    print(np.sum(lb.lb_matrix != mst_symm))
    sw = SashaWang()
    mst_upper = np.triu(mst_symm)
    list_ind = []
    for i in range(mst_symm.shape[0]):
        ind = np.where(mst_upper[i, :] != 0)[0]
        list_ind += zip([i] * len(ind), ind.tolist())
    dist = list(map(lambda x: mst_symm[x[0], x[1]], list_ind))
    dict_vals = dict(zip(list_ind, dist))
    sw.store(dict_vals, mst_symm.shape[0])
    ideal_lb = sw.lb_matrix
    # Floating point error
    assert(np.sum(np.abs(lb.lb_matrix - ideal_lb) > 0.00001) == 0)
    for i in range(size_of_graph):
        for j in range(size_of_graph):
            lower_bound = lb.lb_matrix[i, j]
            assert lower_bound <= out[i][j]

def make_tree(root_index, MST, visited):
    indices = np.where(MST[root_index, :] != 0)[0]
    node = Node(root_index, 0, None, [])
    s = set(indices.tolist()).difference(visited)
    visited |= s
    children = []
    for i in s:
        c = make_tree(i, MST, visited)
        c.parent = node
        c.distance = MST[i, root_index]
        children.append(c)
    node.children = children
    return node

def helper_tri_search_verify(obj):
    global order_val, g, full_mat, debug
    print("PARA TRI SEARCH")
    order = order_val
    val_dict = g
    out = full_mat
    start = time.time()
    obj.store(val_dict, order)
    for i in range(order):
        for j in range(order):
            lb = obj.lookup(i, j)
            assert lb <= out[i][j]
    end = time.time()
    print(end - start)
    for num in range(100):
        i = random.choice(list(range(order)))
        _, neighbours = obj.sparse_matrix.get_row_data(i)
        missing = list(set(list(range(order))).difference(set(neighbours)))
        j = random.choice(missing)
        obj.update([i, j], out[i, j])
        for i in range(order):
            for j in range(order):
                lb = obj.lookup(i, j, fake=True)
                assert lb <= out[i][j]
    end = time.time()
    print(end - start)
    if debug:
        print("Actual values matrix")
        printer_matrix(out)
        print("LB values matrix")
        printer_matrix(obj.lb_matrix)