from algorithms_with_plug_in.prims_with_plug_in import Prims as prims_plugin
from algorithms_with_plug_in.kruskal_with_plug_in import Kruskals as kruskals_plugin
from algorithms_with_plug_in.knn_rp_with_plug_in import kNN_Rp as knnrp
from unified_graph_lb_ub import unified_graph_lb_ub
from sasha_wang import SashaWang
from graph_maker import NlogNGraphMaker
from helper import _get_matrix_from_adj_dict, _bfs
from algorithms_with_plug_in.pam_with_plug_in import PAM as pam_plugin
from algorithms_with_plug_in.clarans_with_plug_in import CLARANS as clarans_plugin
from random import choices, seed, sample
from parametrized_path_search import ParamTriSearch

# Intersection-Trisearch
from IntersctionTriSearch.IntersectTriSearch import IntersectTriSearch
# NodeLandmark
from LAESA import NodeLandMarkRandom
from DSS import DSS
from LSS import LSS

from vanila_algorithms.prims import Prims as vanila_prims
from vanila_algorithms.kruskals import Kruskals as vanila_kruskals
from vanila_algorithms.pam import PAM as vanila_pam
from vanila_algorithms.clarans import CLARANS as vanila_clarans
from vanila_algorithms.knn_rp import kNN_Rp as vanila_knnrp

import numpy as np
import time
import pickle
import os
import pdb
import sys

import copy

debug = False
full_mat = None
count = 0


def oracle(i, j):
    global full_mat, count
    count += 1
    return full_mat[i][j]


def time_waste_oracle(i, j):
    global count
    return oracle(i, j)


def count_reset():
    global count
    count = 0


class Timer:
    def __init__(self):
        self.timer = None
        self.time_elapsed = None

    def start(self):
        self.time_elapsed = 0
        count_reset()
        self.timer = time.time()

    def end(self):
        self.time_elapsed = time.time() - self.timer
        self.timer = 0


def prims_SW(pr, measure, kind, order_val, timer):
    # Sasha Wang algorithm
    global full_mat, count
    
    g = {}
    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()

    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    print("Plugin with Sasha Wang Experiments\nActual(SW) Prims Path Length: {}\nSasha Wang Prims Path Length: {}\n"
          "measure: {}, kind: {}, order_val: {}".
          format(p.mst_path_length, pr.mst_path_length, measure, kind, order_val))
    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(
        timer.time_elapsed - obj_sw.update_time) + "\n"
    print("COUNT Sasha Wang ", count, timer.time_elapsed - obj_sw.update_time, "\n\n")


def prims_DSS(pr, measure, kind, order_val, timer, algo):
    """
    DSS Solution Scheme
    """
    global full_mat, count

    g = {}
    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    print("Plugin with DSS\nActual(SW) {} Path Length: {}\nDSS Path Length: {}\n"
          "measure: {}, kind: {}, order_val: {}".
          format(algo, p.mst_path_length, pr.mst_path_length, measure, kind, order_val))
    print("DSS COUNT {}\nTime: {}\n\n".format(count, timer.time_elapsed, ))


def prims_NodeLandmark(pr, measure, kind, order_val, timer, algo):
    """
        Jees New code the nodeLandmark
        """
    import math
    global full_mat, count

    k_list = [math.ceil(math.log(order_val)) - 2, math.ceil(math.log(order_val)), math.ceil(math.log(order_val)) + 2,
              math.ceil(math.log(order_val)) + 4]
    for k in k_list:
        # g_prime = copy.copy(g)
        g_prime = {}
        timer.start()
        oracle_plugin = NodeLandMarkRandom(g_prime, k, oracle, order_val)
        p = prims_plugin(order_val, oracle, oracle_plugin)
        p.mst(0)
        timer.end()
        print(
            "Node Landmark:\nActual(Vanila) Prims Path Length: {}\nNode Lankmark(our plugin) Prims Path Length: {}\nmeasure: {}, kind: {}, order_val: {}".
                format(pr.mst_path_length, p.mst_path_length, measure, kind, order_val))
        print("COUNT node Landmark {}\nk:{}\nTime: {}\n\n".format(count, k, timer.time_elapsed))


def prims_TSS(pr, measure, kind, order_val, timer, algo):
    """
    Jees added this part to test the naive code for intersection-TriSearch(This)
    This accepts a graph(graph with edge and distance format)
    and converts it into adjacency list representation and stores it.
    This updates a new edge the moment its gets a new resolution
    This computes the value of lower-bound only when needed by a query Q(a, b)
    This find the Triangles through intersection of adjacency list of both end points of the query edge (a & b)
    The class IntersectTriSearch: is initialized with the graph and it takes care of everything else
    including conversion to adjacency list.
    """
    global full_mat, count

    timer.start()
    oracle_plugin = IntersectTriSearch({}, order_val)
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()
    print(
        "IntersctionTriSearch Experiments:\nActual(Vanila) Prims Path Length: {}\nIntersctionTriSearch(our plugin) Prims Path Length: {}\nmeasure: {}, kind: {}, order_val: {}".
            format(pr.mst_path_length, p.mst_path_length, measure, kind, order_val))
    print("COUNT intersct Trisearch: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(count,
                                                                                        timer.time_elapsed - oracle_plugin.sp_time,
                                                                                        oracle_plugin.sp_time))


def prims_LSS(pr, measure, kind, order_val, timer, algo):
    """
    code to test the landmarkbased methods
    """
    global full_mat, count
    timer.start()
    oracle_plugin = LSS({}, order_val, 6, oracle)
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()
    print("{} COUNT LSS: {}\nTime(LSS): {}\n\n".format(algo, count, timer.time_elapsed))


def helper_prims_plugin(g, full_mat1, g_mat, measure, kind, order_val, algo, three=False):
    global full_mat, count
    g, full_mat, g_mat = g, full_mat1, g_mat
    timer = Timer()

    pr = vanila_prims(order_val, time_waste_oracle)
    timer.start()
    pr.mst(0)
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print(base_algo_results, "\n\n")

    prims_SW(pr, measure, kind, order_val, timer)

    prims_DSS(pr, measure, kind, order_val, timer, algo)

    prims_NodeLandmark(pr, measure, kind, order_val, timer, algo)

    prims_TSS(pr, measure, kind, order_val, timer, algo)

    prims_LSS(pr, measure, kind, order_val, timer, algo)

    """
    count = 0
    start = time.time()
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    end = time.time()
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    print("LB Tree - Original Length: {}, our length: {}, measure: {}, kind: {}, order_val: {}".
              format(p.mst_path_length, pr.mst_path_length, measure, kind, order_val))

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))
    print("COUNT LBTree enabled ", count, end - start - obj.update_time, "\n\n")
    lbt_results = "COUNT LBTree enabled " + str(count) + " Time " \
                  + str(end - start - obj.update_time) \
                  + " Average Error with LBT " + str(average_lbt) \
                  + " Average Error with Original " + str(average_original) \
                  + "\n"

    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    end = time.time()

    print("(PRIMS)Tri Search - Original Length: {}, our length: {}, measure: {}, kind: {}, order_val: {}".
              format(p.mst_path_length, pr.mst_path_length, measure, kind, order_val))
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))

    para_results_2 = "PARA 2 " + str(count) + " Time " \
                     + str(end - start - obj.update_time) \
                     + " Average Error with LBT " + str(average_lbt) \
                     + " Average Error with Original " + str(average_original) \
                     + "\n"
    print("PARA ", count, end - start - obj.update_time, "\n\n")


    if three:
        start = time.time()
        count = 0
        obj = ParamTriSearch(3, obj_sw.ub_matrix)
        obj.store(g, order_val)
        oracle_plugin = obj
        p = prims_plugin(order_val, oracle, oracle_plugin)
        p.mst(0)
        end = time.time()

        print("(PRIMS)II Tri Search - Original Length: {}, our lenght: {}, measure: {}, kind: {}, order_val: {}".
                format(p.mst_path_length, pr.mst_path_length, measure, kind, order_val))
        assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001

        average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
        average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))

        para_results_3 = "PARA 3 " + str(count) + " Time " \
                         + str(end - start - obj.update_time) \
                         + " Average Error with LBT " + str(average_lbt) \
                         + " Average Error with Original " + str(average_original) \
                         + "\n"
        print("PARA 3 ", count, end - start - obj.update_time, "\n\n")
    """
    # results_file_name = str(order_val) + "_prims_" + generation_algorithms[kind] + "_" + distance_measure[
    #     measure] + "_" + ".txt"
    # f = open(os.path.join("results", results_file_name), "a+")
    # f.write(base_algo_results)
    # f.write(sasha_wang_results)
    # # f.write(lbt_results)
    # f.write(para_results_2)
    # if three:
    #     f.write(para_results_3)
    # f.write("\n")
    # f.close()


def helper_kruskals_plugin(g, full_mat1, g_mat, measure, kind, order_val, algo, three=False):
    global full_mat, count
    g, full_mat, g_mat = g, full_mat1, g_mat
    timer = Timer()

    pr = vanila_kruskals(order_val, time_waste_oracle)
    timer.start()
    pr.mst()
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("COUNT Without Plugin ", count, timer.time_elapsed, "\n\n")

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    timer.end()

    print("(KRUSKAL)Sasha Wang - Original Length: {}, our lenght: {}, measure: {}, kind: {}, order_val: {}".
          format(pr.mst_path_length, p.mst_path_length, measure, kind, order_val))
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001

    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(timer.time_elapsed - obj_sw.update_time) + "\n"
    print("COUNT Sasha Wang ", count, timer.time_elapsed - obj_sw.update_time, "\n\n")

    """
    DSS Solution Scheme
    """
    g = {}
    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    timer.end()
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    print("Plugin with DSS\nActual(SW) {} Path Length: {}\nDSS Path Length: {}\n"
          "measure: {}, kind: {}, order_val: {}".
          format(algo, p.mst_path_length, pr.mst_path_length, measure, kind, order_val))
    print("DSS COUNT {}\nTime: {}\n\n".format(count, timer.time_elapsed, ))

    """
        Jees added this part to test the naive code for intersection-TriSearch(This)
        This accepts a graph(graph with edge and distance format) 
            and converts it into adjacency list representation and stores it
        This updates a new edge the moment its gets a new resolution
        This computes the value of lower-bound only when needed by a query Q(a, b)
        This find the Triangles through intersection of adjacency list of both end points of the query edge (a & b) 
        The class IntersectTriSearch: is initialized with the graph and it takes care of everything else 
            including conversion to adjacency list. 
    """
    timer.start()
    oracle_plugin = IntersectTriSearch({}, order_val)
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    timer.end()
    print(
        "KRUSKAL - IntersctionTriSearch Experiments:\nActual(Vanila) KRUSKAL Path Length: {}\nIntersctionTriSearch(our plugin) Prims Path Length: {}\nmeasure: {}, kind: {}, order_val: {}".
            format(pr.mst_path_length, p.mst_path_length, measure, kind, order_val))
    print("COUNT intersct Trisearch: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(count,
                                                                                        timer.time_elapsed - oracle_plugin.sp_time,
                                                                                        oracle_plugin.sp_time))
    print("My Lookup count(Tri): {}".format(oracle_plugin.lookup_count))

    """
        code to test the landmarkbased methods
    """

    timer.start()
    oracle_plugin = LSS({}, order_val, 6, oracle)
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    timer.end()
    print("{} COUNT LSS: {}\nTime(LSS): {}\n\n".format(algo, count, timer.time_elapsed))

    """
    start = time.time()
    count = 0
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    end = time.time()
    print("(KRUSKAL) LB Tree - Original Length: {}, our lenght: {}, measure: {}, kind: {}, order_val: {}".
          format(pr.mst_path_length, p.mst_path_length, measure, kind, order_val))
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))
    print("COUNT LBTree enabled ", count, end - start - obj.update_time, "\n\n")
    lbt_results = "COUNT LBTree enabled " + str(count) + " Time " \
                  + str(end - start - obj.update_time) \
                  + " Average Error with LBT " + str(average_lbt) \
                  + " Average Error with Original " + str(average_original) \
                  + "\n"

    count = 0
    start = time.time()
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    end = time.time()
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    print("(KRUSKAL)Tri Search - Original Length: {}, our lenght: {}, measure: {}, kind: {}, order_val: {}".
            format(pr.mst_path_length, p.mst_path_length, measure, kind, order_val))


    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))

    para_results_2 = "PARA 2 " + str(count) + " Time " \
                     + str(end - start - obj.update_time) \
                     + " Average Error with LBT " + str(average_lbt) \
                     + " Average Error with Original " + str(average_original) \
                     + "\n"
    print("PARA ", count, end - start - obj.update_time, "\n\n")

    if three:
        count = 0
        start = time.time()
        obj = ParamTriSearch(3, obj_sw.ub_matrix)
        obj.store(g, order_val)
        oracle_plugin = obj
        p = kruskals_plugin(order_val, oracle, oracle_plugin)
        p.mst()
        end = time.time()
        assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
        print("(KRUSKAL)II Trisearch - Original Length: {}, our lenght: {}, measure: {}, kind: {}, order_val: {}".
              format(pr.mst_path_length, p.mst_path_length, measure, kind, order_val))
        para_results_3 = "PARA 3 " + str(count) + " Time " \
                         + str(end - start - obj.update_time) \
                         + " Average Error with LBT " + str(average_lbt) \
                         + " Average Error with Original " + str(average_original) \
                         + "\n"
        print("PARA 3 ", count, end - start - obj.update_time, "\n\n")
    """
    # results_file_name = str(order_val) + "_kruskals_" + generation_algorithms[kind] + "_" + distance_measure[
    #     measure] + "_" + ".txt"
    # f = open(os.path.join("results", results_file_name), "a+")
    # f.write(base_algo_results)
    # f.write(sasha_wang_results)
    # f.write(lbt_results)
    # f.write(para_results_2)
    # if three:
    #     f.write(para_results_3)
    # f.write("\n")
    # f.close()


def getter(measure, kind, order_val):
    # real_graph_distances_data_20_ForrestFire_64
    distance_measure = ['normal', 'uniform', 'zipf', 'data_flicker', 'data_sf', 'data_20']
    # full_mats = [each + str(order_val) + '.pkl' for each in distance_measure]
    generation_algorithms = ['Geometric', 'Renyi Erdos', 'ForrestFire', 'Barabasi']

    full_mat_name = distance_measure[measure] + str(order_val) + '.pkl'
    g_name = distance_measure[measure] + '_distances_' + generation_algorithms[kind] + '_' + str(order_val) + '.pkl'
    g = pickle.load(open(os.path.join("igraph", g_name), 'rb'))
    full_mat = pickle.load(open(os.path.join("igraph", full_mat_name), 'rb'))
    g_mat = _get_matrix_from_adj_dict(g, order_val)
    return g_mat, g, full_mat


def helper_pam_test(measure, kind, order_val, k):
    from helper_plugins import set_globals, helper_pam_plugin
    _g_mat, _g, _full_mat = getter(measure, kind, order_val)
    set_globals(_g, _g_mat, order_val, _full_mat)
    helper_pam_plugin(k)


def helper_pam_plugin(g, full_mat1, g_mat, measure, kind, order_val, k, algo, three=False):
    global full_mat, count
    centroids = sample(list(range(order_val)), k)
    copy_centroids = copy.copy(centroids)

    g, full_mat, g_mat = g, full_mat1, g_mat
    timer = Timer()

    centroids = copy.copy(copy_centroids)
    timer.start()
    pr = vanila_pam(time_waste_oracle, order_val, k, centroids)
    print("Vanila: ", pr.centroids)
    # oracle, n, k, centroids = None):
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("COUNT Without Plugin ", count, timer.time_elapsed, "\n\n")

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    centroids = copy.copy(copy_centroids)
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    print("Plug-in: ", p.centroids)
    timer.end()

    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(timer.time_elapsed - obj_sw.update_time) + "\n"
    print("COUNT Sasha Wang ", count, timer.time_elapsed - obj_sw.update_time, "\n\n")

    """
    DSS Solution Scheme
    """
    g = {}
    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    timer.end()
    print("DSS: ", p.centroids)
    print("DSS Experiments - PAM\nmeasure: {}, kind: {}, order_val: {}".format(measure, kind, order_val))
    print("DSS COUNT {}\nTime: {}\n\n".format(count, timer.time_elapsed))


    """
        Jees added this part to test the naive code for intersection-TriSearch(This)
        This accepts a graph(graph with edge and distance format) 
            and converts it into adjacency list representation and stores it
        This updates a new edge the moment its gets a new resolution
        This computes the value of lower-bound only when needed by a query Q(a, b)
        This find the Triangles through intersection of adjacency list of both end points of the query edge (a & b) 
        The class IntersectTriSearch: is initialized with the graph and it takes care of everything else 
            including conversion to adjacency list. 
    """
    centroids = copy.copy(copy_centroids)
    timer.start()
    oracle_plugin = IntersectTriSearch({}, order_val)
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    print("TriSearch: ", p.centroids)
    timer.end()
    print(
        "IntersctionTriSearch Experiments - PAM\nmeasure: {}, kind: {}, order_val: {}".format(measure, kind, order_val))
    print("COUNT intersct Trisearch: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(count,
                                                                                        timer.time_elapsed - oracle_plugin.sp_time,
                                                                                        oracle_plugin.sp_time))

    print("*"*40)

    """
    code to test the landmark based methods
    """

    timer.start()
    oracle_plugin = LSS({}, order_val, 6, oracle)
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    timer.end()
    print("LSS: ", p.centroids)
    print("{} COUNT LSS: {}\nTime(LSS): {}\n\n".format(algo, count, timer.time_elapsed))
    """
    centroids = copy.copy(copy_centroids)
    start = time.time()
    count = 0
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    print("Plug-in: ", p.centroids)
    end = time.time()

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))
    print("COUNT LBTree enabled ", count, end - start - obj.update_time, "\n\n")
    lbt_results = "COUNT LBTree enabled " + str(count) + " Time " \
                  + str(end - start - obj.update_time) \
                  + " Average Error with LBT " + str(average_lbt) \
                  + " Average Error with Original " + str(average_original) \
                  + "\n"

    centroids = copy.copy(copy_centroids)
    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    print("Plug-in: ", p.centroids)
    end = time.time()
    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))

    para_results_2 = "PARA 2 " + str(count) + " Time " \
                     + str(end - start - obj.update_time) \
                     + " Average Error with LBT " + str(average_lbt) \
                     + " Average Error with Original " + str(average_original) \
                     + "\n"
    print("PARA ", count, end - start - obj.update_time, "\n\n")

    if three:
        centroids = copy.copy(copy_centroids)
        start = time.time()
        count = 0
        obj = ParamTriSearch(3, obj_sw.ub_matrix)
        obj.store(g, order_val)
        oracle_plugin = obj
        p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
        end = time.time()


        para_results_3 = "PARA 3 " + str(count) + " Time " \
                         + str(end - start - obj.update_time) \
                         + " Average Error with LBT " + str(average_lbt) \
                         + " Average Error with Original " + str(average_original) \
                         + "\n"
        print("PARA 3 ", count, end - start - obj.update_time, "\n\n")

    """
    # results_file_name = str(k) + "_" + str(order_val) + "_pam_" + generation_algorithms[kind] + "_" + distance_measure[
    #     measure] + "_" + ".txt"
    # f = open(os.path.join("results", results_file_name), "a+")
    # f.write(base_algo_results)
    # f.write(sasha_wang_results)
    # # f.write(lbt_results)
    # # f.write(para_results_2)
    # # if three:
    # #     f.write(para_results_3)
    # f.write("\n")
    # f.close()


def helper_clarans_plugin(g, full_mat1, g_mat, measure, kind, order_val, k, algo, three=False):
    global full_mat, count

    g, full_mat, g_mat = g, full_mat1, g_mat
    timer = Timer()

    timer.start()
    pr = vanila_clarans(time_waste_oracle, order_val, k)
    print("Plug-in(CLA): ", pr.centroids)
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("{} COUNT Without Plugin {} Time: {}\n\n".format(algo, count, timer.time_elapsed))

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    print("Plug-in(CLA): ", p.centroids)
    timer.end()

    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(timer.time_elapsed - obj_sw.update_time) + "\n"
    print("COUNT Sasha Wang ", count, timer.time_elapsed - obj_sw.update_time, "\n\n")

    """
    DSS Solution Scheme
    """
    g = {}
    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    timer.end()
    print("DSS: ", p.centroids)
    print("DSS Experiments - PAM\nmeasure: {}, kind: {}, order_val: {}".format(measure, kind, order_val))
    print("{} DSS COUNT {}\nTime: {}\n\n".format(algo, count, timer.time_elapsed))

    """
        Jees added this part to test the naive code for intersection-TriSearch(This)
        This accepts a graph(graph with edge and distance format) 
            and converts it into adjacency list representation and stores it
        This updates a new edge the moment its gets a new resolution
        This computes the value of lower-bound only when needed by a query Q(a, b)
        This find the Triangles through intersection of adjacency list of both end points of the query edge (a & b) 
        The class IntersectTriSearch: is initialized with the graph and it takes care of everything else 
            including conversion to adjacency list. 
    """
    timer.start()
    oracle_plugin = IntersectTriSearch({}, order_val)
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    print("TriSearch: ", p.centroids)
    timer.end()
    print(
        "IntersctionTriSearch Experiments - CLARANS\nmeasure: {}, kind: {}, order_val: {}".format(measure, kind,
                                                                                                  order_val))
    print("COUNT intersct Trisearch: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(count,
                                                                                        timer.time_elapsed - oracle_plugin.sp_time,
                                                                                        oracle_plugin.sp_time))

    print("*"*40)

    """
    code to test the landmarkbased methods
    """

    timer.start()
    oracle_plugin = LSS({}, order_val, 6, oracle)
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    print("LSS: ", p.centroids)
    timer.end()
    print("{} COUNT LSS: {}\nTime(LSS): {}\n\n".format(algo, count, timer.time_elapsed))

    """
    start = time.time()
    count = 0
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    print("Plug-in(CLA): ", p.centroids)
    end = time.time()

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))
    print("COUNT LBTree enabled ", count, end - start - obj.update_time, "\n\n")
    lbt_results = "COUNT LBTree enabled " + str(count) + " Time " \
                  + str(end - start - obj.update_time) \
                  + " Average Error with LBT " + str(average_lbt) \
                  + " Average Error with Original " + str(average_original) \
                  + "\n"

    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = clarans_plugin(order_val, k, oracle_plugin, oracle)
    print("Plug-in(CLA): ", p.centroids)
    end = time.time()

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))

    para_results_2 = "PARA 2 " + str(count) + " Time " \
                     + str(end - start - obj.update_time) \
                     + " Average Error with LBT " + str(average_lbt) \
                     + " Average Error with Original " + str(average_original) \
                     + "\n"
    print("PARA ", count, end - start - obj.update_time, "\n\n")



    if three:
        start = time.time()
        count = 0
        obj = ParamTriSearch(3, obj_sw.ub_matrix)
        obj.store(g, order_val)
        oracle_plugin = obj
        p = clarans_plugin(order_val, k, oracle_plugin, oracle)
        end = time.time()
        para_results_3 = "PARA 3 " + str(count) + " Time " \
                         + str(end - start - obj.update_time) \
                         + " Average Error with LBT " + str(average_lbt) \
                         + " Average Error with Original " + str(average_original) \
                         + "\n"
        print("PARA 3 ", count, end - start - obj.update_time, "\n\n")
    """
    # results_file_name = str(k) + "_" + str(order_val) + "_clarans_" + generation_algorithms[kind] + "_" + \
    #                     distance_measure[
    #                         measure] + "_" + ".txt"
    # f = open(os.path.join("results", results_file_name), "a+")
    # f.write(base_algo_results)
    # f.write(sasha_wang_results)
    # # f.write(lbt_results)
    # # f.write(para_results_2)
    # # if three:
    # #     f.write(para_results_3)
    # f.write("\n")
    # f.close()


def helper_navarro_plugin(g, full_mat1, g_mat, measure, kind, order_val, k, algo, three=False):
    global full_mat, count

    g, full_mat, g_mat = g, full_mat1, g_mat
    timer = Timer()

    timer.start()
    pr = vanila_knnrp(order_val, k, oracle)
    pr.knn_queries()
    # print("Plug-in(Nav): ", pr.NHA)
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("COUNT Without Plugin ", count, timer.time_elapsed, "\n\n")

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = knnrp(order_val, k, oracle, oracle_plugin)
    p.knn_queries()
    # print("Plug-in(Nav): ", p.NHA)
    timer.end()

    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(timer.time_elapsed - obj_sw.update_time) + "\n"
    print("COUNT Sasha Wang ", count, timer.time_elapsed - obj_sw.update_time, "\n\n")

    """
    DSS Solution Scheme
    """
    g = {}
    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = knnrp(order_val, k, oracle, oracle_plugin)
    p.knn_queries()
    timer.end()
    print("DSS Experiments - {}\nmeasure: {}, kind: {}, order_val: {}".format(algo, measure, kind, order_val))
    print("{} DSS COUNT {}\nTime: {}\n\n".format(algo, count, timer.time_elapsed))

    """
        Jees added this part to test the naive code for intersection-TriSearch(This)
        This accepts a graph(graph with edge and distance format) 
            and converts it into adjacency list representation and stores it
        This updates a new edge the moment its gets a new resolution
        This computes the value of lower-bound only when needed by a query Q(a, b)
        This find the Triangles through intersection of adjacency list of both end points of the query edge (a & b) 
        The class IntersectTriSearch: is initialized with the graph and it takes care of everything else 
            including conversion to adjacency list. 
    """

    timer.start()
    oracle_plugin = IntersectTriSearch({}, order_val)
    p = knnrp(order_val, k, oracle, oracle_plugin)
    p.knn_queries()
    # print("Plug-in(Nav): ", p.NHA)
    timer.end()
    print("IntersctionTriSearch Experiments - {}\nmeasure: {}, kind: {}, order_val: {}".format(algo, measure, kind,
                                                                                                   order_val))
    print("{} COUNT intersct Trisearch: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(algo, count,
                                                                                        timer.time_elapsed - oracle_plugin.sp_time,
                                                                                        oracle_plugin.sp_time))
    print("*"*40)

    """
    code to test the landmark based methods
    """

    timer.start()
    oracle_plugin = LSS({}, order_val, 6, oracle)
    p = knnrp(order_val, k, oracle, oracle_plugin)
    p.knn_queries()
    timer.end()
    print("COUNT LSS: {}\nTime(LSS): {}\n\n".format(count, timer.time_elapsed))

    """
    start = time.time()
    count = 0
    obj = unified_graph_lb_ub()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = knnrp(order_val, k, oracle, oracle_plugin)

    p.knn_queries()
    print("Plug-in(Nav): ", p.NHA)
    end = time.time()

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))
    print("COUNT LBTree enabled ", count, end - start - obj.update_time, "\n\n")
    lbt_results = "COUNT LBTree enabled " + str(count) + " Time " \
                  + str(end - start - obj.update_time) \
                  + " Average Error with LBT " + str(average_lbt) \
                  + " Average Error with Original " + str(average_original) \
                  + "\n"

    start = time.time()
    count = 0
    obj = ParamTriSearch(2, obj_sw.ub_matrix)
    obj.store(g, order_val)
    oracle_plugin = obj
    p = knnrp(order_val, k, oracle, oracle_plugin)

    p.knn_queries()
    print("Plug-in(Nav): ", p.NHA)
    end = time.time()

    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))

    para_results_2 = "PARA 2 " + str(count) + " Time " \
                     + str(end - start - obj.update_time) \
                     + " Average Error with LBT " + str(average_lbt) \
                     + " Average Error with Original " + str(average_original) \
                     + "\n"
    print("PARA ", count, end - start - obj.update_time, "\n\n")



    if three:
        start = time.time()
        count = 0
        obj = ParamTriSearch(3, obj_sw.ub_matrix)
        obj.store(g, order_val)
        oracle_plugin = obj
        p = knnrp(order_val, k, oracle, oracle_plugin)
        p.knn_queries()
        end = time.time()
        para_results_3 = "PARA 3 " + str(count) + " Time " \
                         + str(end - start - obj.update_time) \
                         + " Average Error with LBT " + str(average_lbt) \
                         + " Average Error with Original " + str(average_original) \
                         + "\n"
        print("PARA 3 ", count, end - start - obj.update_time, "\n\n")
    """
    # results_file_name = str(k) + "_" + str(order_val) + "_navarro_" + generation_algorithms[kind] + "_" + \
    #                     distance_measure[
    #                         measure] + "_" + ".txt"
    # f = open(os.path.join("results", results_file_name), "a+")
    # f.write(base_algo_results)
    # f.write(sasha_wang_results)
    # # f.write(lbt_results)
    # # f.write(para_results_2)
    # # if three:
    # #     f.write(para_results_3)
    # f.write("\n")
    # f.close()
