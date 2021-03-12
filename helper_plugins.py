from algorithms_with_plug_in.prims_with_plug_in import Prims as prims_plugin
from algorithms_with_plug_in.kruskal_with_plug_in import Kruskals as kruskals_plugin
from unified_graph_lb_ub import unified_graph_lb_ub
from bottom_up_tree_sample import BottomUPTree as bu_tree
from bottom_up_tree_sample_min_max import BottomUPTree as bu_tree_min_max
from probablistic_bottom_up_tree_sampling import BottomUPTree as bu_tree_prob
from sasha_wang import SashaWang
from graph_maker import NlogNGraphMaker
from helper import _get_matrix_from_adj_dict, _bfs
from algorithms_with_plug_in.pam_with_plug_in import PAM as pam_plugin
from vanila_algorithms.pam import PAM as pam_vanila
from algorithms_with_plug_in.clarans_with_plug_in import CLARANS as clarans_plugin
from vanila_algorithms.clarans import CLARANS as clarans_vanila
from random import sample, seed
from parametrized_path_search import ParamTriSearch
import numpy as np
import time
import copy
from lower_bound_tree import LBTree

debug = False
order_val = 128
graph_maker = NlogNGraphMaker(order_val)
g = graph_maker.get_nlogn_edges()
full_mat = graph_maker.out
g_mat = _get_matrix_from_adj_dict(g, order_val)
_bfs(0, g, order_val)
count = 0


def compare_bu_tree():
    global order_val
    for i in range(20):
        graph_maker_local = NlogNGraphMaker(order_val)
        g_local = graph_maker_local.get_nlogn_edges()
        full_mat_local = graph_maker_local.out
        g_mat_local = _get_matrix_from_adj_dict(g_local, order_val)
        lb_min_max = np.copy(g_mat_local)
        start = time.time()
        min_max_tree = bu_tree_min_max(g_local, order_val)
        lb = LBTree(min_max_tree.tree, lb_min_max)
        end = time.time()
        min_max_updates = np.sum(np.abs(lb.lb_matrix - g_mat_local)>0.00001)
        print("Min max ", min_max_updates, end-start)
        lb_min = np.copy(g_mat_local)
        start = time.time()
        min_max_tree = bu_tree(g_local, order_val)
        lb = LBTree(min_max_tree.tree, lb_min)
        end = time.time()
        min_updates = np.sum(np.abs(lb.lb_matrix - g_mat_local)>0.00001)
        print("Min ", min_updates, end-start)
        lb_min = np.copy(g_mat_local)
        start = time.time()
        prob_tree = bu_tree_prob(g_local, order_val)
        lb = LBTree(prob_tree.tree, lb_min)
        end = time.time()
        prob_updates = np.sum(np.abs(lb.lb_matrix - g_mat_local)>0.00001)
        print("Probablistic ", prob_updates, end-start)
        print("Ratio :",min_max_updates/min_updates, " min max : ", min_max_updates, " min : ", min_updates)

def oracle(i, j):
    global full_mat, count
    count += 1
    return full_mat[i][j]


def helper_prims_plugin():
    global g, g_mat, order_val, full_mat, count
    count = 0
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    assert not np.any(np.abs(np.array(obj_sw.lb_matrix)-np.array(obj.lb_matrix))<-0.0000001)
    start = time.time()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    end = time.time()
    print("COUNT Sasha Wang", count, end - start,"\n\n")

    count = 0
    start = time.time()
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    end = time.time()
    print("COUNT LBTree enabled", count, end - start, "\n\n")

    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    end = time.time()
    print("PARA", count, end - start, "\n\n")


def helper_kruskals_plugin():
    global g, g_mat, order_val, full_mat, count
    count = 0
    start = time.time()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    end = time.time()
    print('COUNT Sasha Wang', count, end - start, "\n\n")

    start = time.time()
    count = 0
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    end = time.time()
    print("COUNT LBTree enabled", count, end - start, "\n\n")

    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    end = time.time()
    print("PARA", count, end - start, "\n\n")

def set_globals(_g, _g_mat, _order_val, _full_mat):
    global g, g_mat, order_val, full_mat
    g, g_mat, order_val, full_mat = _g, _g_mat, _order_val, _full_mat


def helper_pam_plugin(k=5):
    global g, g_mat, order_val, full_mat, count
    centroids = sample(list(range(order_val)), k)
    copy_centroids = copy.copy(centroids)

    count = 0
    centroids = copy.copy(copy_centroids)
    start = time.time()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    end = time.time()
    print("SW", p.centroids)
    print("COUNT Sasha Wang", count, end - start)

    centroids = copy.copy(copy_centroids)
    start = time.time()
    count = 0
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    end = time.time()
    print("LBT", p.centroids)
    print("COUNT LBTree enabled", count, end - start, "\n\n")

    centroids = copy.copy(copy_centroids)
    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    end = time.time()
    print("PARA", p.centroids)
    print("PARA", count, end - start, "\n\n")
    centroids = copy.copy(copy_centroids)
    p = pam_vanila(oracle, order_val, k, centroids)
    print("Original: ", p.centroids)


def helper_clarans_plugin():
    # n, k, plug_in_oracle, oracle, num_local=None, max_neighbour=None)
    global g, g_mat, order_val, full_mat, count
    k = 5
    pr = clarans_vanila(oracle, order_val, k)
    print("ACTUAL: ", pr.centroids)
    count = 0
    start = time.time()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    end = time.time()
    print("SW", p.centroids)
    print("COUNT Sasha Wang", count, end - start)

    start = time.time()
    count = 0
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    end = time.time()
    print("LBT", p.centroids)
    print("COUNT LBTree enabled", count, end - start, "\n\n")

    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    end = time.time()
    print("PARA", p.centroids)
    print("PARA", count, end - start, "\n\n")
