from algorithms_with_plug_in.prims_with_plug_in import Prims as prims_plugin
# from algorithms_with_plug_in.kruskal_with_plug_in import Kruskals as kruskals_plugin
from algorithms_with_plug_in.kruskal_with_plug_in_new import Kruskals as kruskals_plugin
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

from dataprep_ficker1M.cure_1M_to_32K import get_oracle
from SF_POI.SF_Oracle import get_SF_ORACLE

import numpy as np
import time
import pickle
import os
import pdb
import sys
import math
from datetime import datetime

import copy

debug = False
full_mat = None
count = 0


def oracle(i, j):
    global full_mat, count
    count += 1
    return full_mat[i][j]


def time_waste_oracle(i, j):
    return oracle(i, j)


def update_count():
    global count
    count += 1


def count_reset():
    global count
    count = 0


def get_real_counter_oracle(o):
    graph = dict()

    def oracle(u, v):
        if u == v:
            return 0
        x, y = min(u, v), max(u, v)
        if (x, y) not in graph:
            graph[(x, y)] = o(u, v)
        return graph[(x, y)]

    return oracle


def UCIUrbanOracle(debug=False):
    print("In UCI Urban Oracle")
    o = get_SF_ORACLE(name="urbanGB.txt", limit_rows=17000, maximum=20016, debug=debug)
    def oracle(u, v):
        update_count()
        return o(u, v)
    return oracle


def SFOracle(debug=False):
    print("In SF Oracle")
    o = get_SF_ORACLE(name="CA.txt", limit_rows=17000, maximum=1201, debug=debug)

    def oracle(u, v):
        update_count()
        return o(u, v)

    return oracle


def flicker_oracle():
    print("In Flicker Oracle")
    o = get_oracle()

    def oracle(u, v):
        update_count()
        return o(u, v)

    return oracle


def date_time_printer():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string, "\n")


class SW_and_LBUB_Oracle:
    def __init__(self):
        self.obj_sw = SashaWang()
        self.obj_lbub = unified_graph_lb_ub()
        self.is_uncalculated = self.obj_sw.is_uncalculated

    def store(self, graph, order_val):
        self.G = copy.copy(graph)
        self.obj_sw.store(graph, order_val)
        self.obj_lbub.store(self.G, order_val)

    def update(self, edge, val):
        self.obj_sw.update(edge, val)
        self.obj_lbub.update(edge, val)

    def lookup(self, x, y):
        sw_lb_ub = self.obj_sw.lookup(x, y)
        lbub_lb_ub = self.obj_lbub.lookup(x, y)
        if (lbub_lb_ub[0] - sw_lb_ub[0]) > 0.000000001:
            print("Found a better lower bound for {}, {}".format(x, y))
        if (- lbub_lb_ub[1] + sw_lb_ub[1]) > 0.000000001:
            print("Found a better upper bound for {}, {}".format(x, y))
        return lbub_lb_ub


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


def prims_SW(g, pr, order_val, timer, oracle, bounder=False):
    # Sasha Wang algorithm
    print("Experiment Starting Sasha Wang (Prims)\n")

    global full_mat, count
    # g = {}

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    # oracle = flicker_oracle()
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()

    print("value from SW: {}".format(p.mst_path_length))
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    print("Plugin with Sasha Wang Experiments\nActual(SW) Prims Path Length: {}\nSasha Wang Prims Path Length: {}\n"
          "order_val: {}".
          format(p.mst_path_length, pr.mst_path_length, order_val))
    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(
        timer.time_elapsed - obj_sw.update_time) + "\n"
    print(
        "COUNT Sasha Wang: {}, Time(total): {}, Time(SP): {}\n\n".format(count, timer.time_elapsed, obj_sw.update_time))

    if bounder:
        lb = []
        ub = []
        lb_name = 'lower_bounds_{}_sw.lb'.format(order_val)
        ub_name = 'upper_bounds_{}_sw.ub'.format(order_val)
        for i in range(order_val):
            for j in range(i+1, order_val):
                a, b = p.plug_in_oracle.lookup(i, j)
                lb.append(a)
                ub.append(b)
        path_out = os.path.join(os.getcwd(), "bounds_compare_results", "bounds_{}_sw".format(order_val))
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        with open(os.path.join(path_out, lb_name), 'w') as f:
            f.write('\n'.join([str(b) for b in lb]))
        with open(os.path.join(path_out, ub_name), 'w') as f:
            f.write('\n'.join([str(b) for b in ub]))


def prims_DSS(g, pr, order_val, timer, oracle, bounder=False):
    """
    DSS Solution Scheme
    """
    print("Experiment Starting DSS (Prims)\n")

    global full_mat, count
    # g = {}
    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    print("Plugin with DSS\nActual(SW) Path Length: {}\nDSS Path Length: {}\n"
          "order_val: {}".
          format(p.mst_path_length, pr.mst_path_length, order_val))
    print("DSS COUNT {}\nTime: {}\n\n".format(count, timer.time_elapsed, ))

    if bounder:
        lb = []
        ub = []
        lb_name = 'lower_bounds_{}_dss.lb'.format(order_val)
        ub_name = 'upper_bounds_{}_dss.ub'.format(order_val)
        for i in range(order_val):
            for j in range(i+1, order_val):
                a, b = p.plug_in_oracle.lookup(i, j)
                lb.append(a)
                ub.append(b)
        path_out = os.path.join(os.getcwd(), "bounds_compare_results", "bounds_{}_dss".format(order_val))
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        with open(os.path.join(path_out, lb_name), 'w') as f:
            f.write('\n'.join([str(b) for b in lb]))
        with open(os.path.join(path_out, ub_name), 'w') as f:
            f.write('\n'.join([str(b) for b in ub]))


def prims_NodeLandmark(g, order_val, timer, scale_k, oracle, bounder=False):
    """
    Jees New code the nodeLandmark
    """
    print("Experiment Starting NodeLandMark (Prims)\n")

    import math
    global full_mat, count

    # k_list = [math.ceil(math.log(order_val)) - 2, math.ceil(math.log(order_val)), math.ceil(math.log(order_val)) + 2,
    # math.ceil(math.log(order_val)) + 4]
    v = math.ceil(math.log(order_val, 2))
    k_list = [v * scale_k, v * scale_k * 2, v * scale_k * 3, v * scale_k * 4, v * scale_k * 5, v * scale_k * 6]
    k_list = k_list[:1]
    print("Value of K: {}, Scale Factor: {}".format(k_list, scale_k))
    # oracle = flicker_oracle()
    for k in k_list:
        # g_prime = copy.copy(g)
        g_prime = g
        g = {}
        print("Number of Nodes in the start of our the LAESA: {}".format(len(g_prime)))
        timer.start()
        oracle_plugin = NodeLandMarkRandom(g_prime, k, oracle, order_val)
        p = prims_plugin(order_val, oracle, oracle_plugin)
        p.mst(0)
        timer.end()
        print(
            "Node Landmark:\nNode Lankmark(our plugin) Prims Path Length: {}\norder_val: {}".
                format(p.mst_path_length, order_val))
        print("COUNT node Landmark {}\nk:{}\nTime: {}\n\n".format(count, k, timer.time_elapsed))
        if bounder:
            lb = []
            ub = []
            lb_name = 'lower_bounds_{}_laesa.lb'.format(order_val)
            ub_name = 'upper_bounds_{}_laesa.ub'.format(order_val)
            for i in range(order_val):
                for j in range(i + 1, order_val):
                    a, b = p.plug_in_oracle.lookup(i, j)
                    lb.append(a)
                    ub.append(b)
            path_out = os.path.join(os.getcwd(), "bounds_compare_results", "bounds_{}_laesa".format(order_val))
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            with open(os.path.join(path_out, lb_name), 'w') as f:
                f.write('\n'.join([str(b) for b in lb]))
            with open(os.path.join(path_out, ub_name), 'w') as f:
                f.write('\n'.join([str(b) for b in ub]))


def prims_TSS(g, order_val, timer, k, oracle, bounder=False):
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
    print("Experiment Starting TSS (Prims)\n")
    global full_mat, count
    # g = {}
    # oracle = flicker_oracle()
    timer.start()
    oracle_plugin = IntersectTriSearch(g, order_val)
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()
    print(
        "Intersect Trisearch Experiments:\nIntersctionTriSearch(our plugin) Prims Path Length: {},\norder_val: {}\n# "
        "of Landmarks: {}".
            format(p.mst_path_length, order_val, k))
    print("COUNT intersect Trisearch: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(count,
                                                                                         timer.time_elapsed - oracle_plugin.sp_time,
                                                                                         oracle_plugin.sp_time))
    if bounder:
        lb = []
        ub = []
        lb_name = 'lower_bounds_{}_tss.lb'.format(order_val)
        ub_name = 'upper_bounds_{}_tss.ub'.format(order_val)
        for i in range(order_val):
            for j in range(i+1, order_val):
                a, b = p.plug_in_oracle.lookup(i, j)
                lb.append(a)
                ub.append(b)
        path_out = os.path.join(os.getcwd(), "bounds_compare_results", "bounds_{}_tss".format(order_val))
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        with open(os.path.join(path_out, lb_name), 'w') as f:
            f.write('\n'.join([str(b) for b in lb]))
        with open(os.path.join(path_out, ub_name), 'w') as f:
            f.write('\n'.join([str(b) for b in ub]))


def prims_LSS(g, order_val, timer, oracle, k):
    """
    code to test the landmarkbased methods
    """
    print("Experiment Starting LSS (Prims)\n")

    global full_mat, count
    # g = {}
    k_ = math.ceil(math.log(order_val, 2))
    timer.start()
    oracle_plugin = LSS(g, order_val, k_, oracle)
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()
    print("COUNT LSS: {}\nTime(LSS): {}\nOrder_val: {}\nLandmarks In original graph:{}\n\n".
          format(count, timer.time_elapsed, order_val, k))
    print("Total Lookups: {}\nUpper Beats: {}\nLower Beats: {}".format(oracle_plugin.lookup_ctr,
                                                                       oracle_plugin.ub_better_ctr,
                                                                       oracle_plugin.lb_better_ctr))


def prims_LBUB(g, pr, measure, kind, order_val, timer, algo):
    timer.start()
    obj = unified_graph_lb_ub()
    # obj = SW_and_LBUB_Oracle()
    obj.store(g, order_val)
    oracle_plugin = obj
    p = prims_plugin(order_val, oracle, oracle_plugin)
    p.mst(0)
    timer.end()
    print("LB Tree - Original Length: {}, our length: {}, measure: {}, kind: {}, order_val: {}".
          format(p.mst_path_length, pr.mst_path_length, measure, kind, order_val))

    # average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    # average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))
    print("COUNT LBTree enabled {}\nTotal Time: {}\n\n\n".format(count, timer.time_elapsed))


def helper_prims_plugin(order_val, scheme_chooser_list, scale_k, oracle_chooser, bounder=False, three=False):
    global count

    # Choosing the right oracle
    oracle = None
    if oracle_chooser == 1:
        oracle = flicker_oracle()
    elif oracle_chooser == 2:
        oracle = SFOracle(debug=False)
    elif oracle_chooser == 3:
        oracle = UCIUrbanOracle(debug=False)
        # oracle = SFOracle(debug=True)
    assert oracle is not None

    timer = Timer()
    print(math.ceil(math.log(order_val, 2)))
    k = math.ceil(math.log(order_val, 2)) * scale_k
    print("Value of K: {}, Scale Factor: {}".format(k, scale_k))

    oracle_plugin = NodeLandMarkRandom({}, k, oracle, order_val)
    print("Priming Cost: {}\nTime for priming: {}\n".format(count, oracle_plugin.time2prime))

    # pr = vanila_prims(order_val, time_waste_oracle)
    pr = vanila_prims(order_val, oracle)
    timer.start()
    pr.mst(0)
    timer.end()

    base_algo_results = "COUNT Without Plugin(Vanila) " + str(count) + "\nTime(Vanila) " + str(
        timer.time_elapsed) + "\n"
    print(base_algo_results, "\n\n")

    bounder = False
    if "SW" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        prims_SW(g, pr, order_val, timer, oracle, bounder=bounder)
        date_time_printer()
    if "TSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        print("The graph has total this many edges at start of TSS: {}".format(len(g)))
        prims_TSS(g, order_val, timer, k, oracle, bounder=bounder)
        date_time_printer()
    if "DSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        prims_DSS(g, pr, order_val, timer,oracle,  bounder=bounder)
        date_time_printer()
    if "NLM" in scheme_chooser_list:
        g = {}
        print("The graph has total this many edges at start of NLM: {}".format(len(g)))
        prims_NodeLandmark(g, order_val, timer, scale_k, oracle, bounder=bounder)
        date_time_printer()
    if "LSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        print("The graph has total this many edges at start of LSS: {}".format(len(g)))
        prims_LSS(g, order_val, timer, oracle, k)
        date_time_printer()
    if "LBUB" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        prims_LBUB(g, pr, order_val, timer)
        date_time_printer()

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


# def kruskals_SW(pr, measure, kind, order_val, timer, algo):
def kruskals_SW(g, pr, order_val, timer, oracle):
    print("Experiment Starting SW (Kruskals)\n")
    global full_mat, count
    # g = {}

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    timer.end()

    print("(KRUSKAL)Sasha Wang - Original Length: {}, our lenght: {}, order_val: {}".
          format(pr.mst_path_length, p.mst_path_length, order_val))
    assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001

    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(
        timer.time_elapsed - obj_sw.update_time) + "\n"
    print(
        "COUNT Sasha Wang: {}, Time(total): {}, Time(SP): {}\n\n".format(count, timer.time_elapsed, obj_sw.update_time))


def kruskals_DSS(g, pr, order_val, timer, oracle):
    """
    DSS Solution Scheme
    """
    print("Experiment Starting DSS (Kruskals)\n")

    global full_mat, count
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
          format("DSS", p.mst_path_length, pr.mst_path_length, order_val))
    print("DSS COUNT {}\nTime: {}\n\n".format(count, timer.time_elapsed, ))


def kruskals_TSS(g, pr, order_val, timer, oracle):
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
    print("Experiment Starting TSS (Kruskals)\n")

    global full_mat, count
    g = g
    # g = {}

    timer.start()
    oracle_plugin = IntersectTriSearch(g, order_val)
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    timer.end()
    print(
        "KRUSKAL - IntersctionTriSearch Experiments:\nActual(Vanila) KRUSKAL Path Length: {}\nIntersctionTriSearch(our plugin) Kruskals Path Length: {}\n, order_val: {}".
            format(pr.mst_path_length, p.mst_path_length, order_val))
    print("COUNT intersct Trisearch: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(count,
                                                                                        timer.time_elapsed - oracle_plugin.sp_time,
                                                                                        oracle_plugin.sp_time))
    print("My Lookup count(Tri): {}".format(oracle_plugin.lookup_count))


def kruskals_LSS(g, pr, order_val, timer, oracle):
    """
    code to test the landmark based methods
    """
    print("Experiment Starting LSS (Kruskals)\n")

    global full_mat, count
    g = {}

    timer.start()
    oracle_plugin = LSS(g, order_val, 6, oracle)
    p = kruskals_plugin(order_val, oracle, oracle_plugin)
    p.mst()
    timer.end()
    print("{} COUNT LSS: {}\nTime(LSS): {}\n\n".format("LSS", count, timer.time_elapsed))

def Kruskals_NodeLandmark(g, order_val, timer, scale_k, oracle):
    """
    Jees New code the nodeLandmark
    """
    print("Experiment Starting NodeLandMark (Kruskals)\n")

    import math
    global full_mat, count

    # k_list = [math.ceil(math.log(order_val)) - 2, math.ceil(math.log(order_val)), math.ceil(math.log(order_val)) + 2,
    # math.ceil(math.log(order_val)) + 4]
    v = math.ceil(math.log(order_val, 2))
    k_list = [v * scale_k, v * scale_k * 2, v * scale_k * 3, v * scale_k * 4, v * scale_k * 5, v * scale_k * 6]
    k_list = k_list[:1]
    print("Value of K: {}, Scale Factor: {}".format(k_list, scale_k))
    # oracle = flicker_oracle()
    for k in k_list:
        # g_prime = copy.copy(g)

        g = {}
        g_prime = g
        print("Number of Nodes in the start of our the LAESA: {}".format(len(g_prime)))
        timer.start()
        oracle_plugin = NodeLandMarkRandom(g_prime, k, oracle, order_val)
        p = kruskals_plugin(order_val, oracle, oracle_plugin)
        p.mst()
        timer.end()
        print(
            "Node Landmark:\nNode Lankmark(our plugin) Kruskals Path Length: {}\norder_val: {}".
                format(p.mst_path_length, order_val))
        print("COUNT node Landmark {}\nk:{}\nTime: {}\n\n".format(count, k, timer.time_elapsed))



def helper_kruskals_plugin(order_val, scheme_chooser_list, scale_k, oracle_chooser, three=False):
    # helper_kruskals_plugin()
    global full_mat, count

    # Choosing the right oracle
    oracle = None
    if oracle_chooser == 1:
        oracle = flicker_oracle()
    elif oracle_chooser == 2:
        oracle = SFOracle(debug=False)
    elif oracle_chooser == 3:
        oracle = UCIUrbanOracle(debug=False)
        # oracle = SFOracle(debug=True)
    assert oracle is not None

    print(math.ceil(math.log(order_val, 2)))
    k = math.ceil(math.log(order_val, 2)) * scale_k
    print("Value of K: {}, Scale Factor: {}".format(k, scale_k))

    oracle_plugin = NodeLandMarkRandom({}, k, oracle, order_val)
    print("Priming Cost: {}\nTime for priming: {}\n".format(count, oracle_plugin.time2prime))


    # g, full_mat, g_mat = g, full_mat1, g_mat
    timer = Timer()

    pr = vanila_kruskals(order_val, oracle)
    timer.start()
    pr.mst()
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("COUNT Without Plugin ", count, timer.time_elapsed, "\n\n")

    if "SW" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        # kruskals_SW(pr, measure, kind, order_val, timer, algo)
        kruskals_SW(g_prime, pr, order_val, timer, oracle)
        date_time_printer()

    if "DSS" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        kruskals_DSS(g_prime, pr, order_val, timer, oracle)
        date_time_printer()

    if "TSS" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        kruskals_TSS(g_prime, pr, order_val, timer, oracle)
        date_time_printer()

    if "LSS" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        kruskals_LSS(g_prime, pr, order_val, timer, oracle)
        date_time_printer()

    if "NLM" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        print("The graph has total this many edges at start of NLM: {}".format(len(g_prime)))
        Kruskals_NodeLandmark(g_prime, order_val, timer, scale_k, oracle)
        date_time_printer()

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


# def pam_SW(pr, measure, kind, order_val, timer, algo, copy_centroids, k):
def pam_SW(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, verbose=False):
    print("Experiment Starting SW (PAM)\n")

    global full_mat, count
    centroids = copy.copy(copy_centroids)
    g = g_prime
    # g = {}
    print("The SW for PAM is starting with {} edges in graph.".format(len(g)))
    print("PAM SW Number of Clusters Chosen: {}".format(k))

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    timer.end()
    if verbose:
        print("Plug-in: ", p.centroids)

    sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(
        timer.time_elapsed - obj_sw.update_time) + "\n"
    print(
        "COUNT Sasha Wang: {}\nTotal Count ( TSS + Priming ): {}\nTime(total): {}, Time(SP): {}\n\n".format(count, priming_cost+count, timer.time_elapsed, obj_sw.update_time))


# def pam_DSS(pr, measure, kind, order_val, timer, algo, copy_centroids, k):
def pam_DSS(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, verbose=False):
    """
    DSS Solution Scheme
    """
    print("Experiment Starting DSS (PAM)\n")

    global full_mat, count
    centroids = copy.copy(copy_centroids)
    g = g_prime
    # g = {}
    print("The DSS for PAM is starting with {} edges in graph.".format(len(g)))
    print("PAM DSS Number of Clusters Chosen: {}".format(k))

    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    timer.end()
    if verbose:
        print("DSS: ", p.centroids)
    print("DSS Experiments - PAM\n, order_val: {}".format(order_val))
    print("DSS COUNT {}\nTime: {}\n\n".format(count, timer.time_elapsed))
    print("Total DSS + Prime: {}".format(count+priming_cost))



# def pam_TSS(pr, measure, kind, order_val, timer, algo, copy_centroids, k):
def pam_TSS(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, verbose=False):
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
    print("Experiment Starting TSS (PAM)\n")

    global full_mat, count
    centroids = copy.copy(copy_centroids)
    g = g_prime
    g = {}
    print("The TSS for PAM is starting with {} edges in graph.".format(len(g)))
    print("PAM TSS Number of Clusters Chosen: {}".format(k))

    timer.start()
    oracle_plugin = IntersectTriSearch(g, order_val)
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)

    timer.end()
    if verbose:
        print("TriSearch: ", p.centroids)
    print(
        "TSS Experiments - PAM\n order_val: {}".format(order_val))
    print("COUNT TSS: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".format(count,timer.time_elapsed - oracle_plugin.sp_time,
                                                                        oracle_plugin.sp_time))
    print("Total TSS + Prime: {}".format(count + priming_cost))
    print("*" * 40)


# def pam_LSS(pr, measure, kind, order_val, timer, algo, copy_centroids, k):
def pam_LSS(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, prime_nodes, prime_edges, verbose=False):
    """
    code to test the landmark based methods
    """
    print("Experiment Starting LSS (PAM)\n")

    global full_mat, count

    centroids = copy.copy(copy_centroids)
    g = g_prime
    # g = {}
    print("The LSS for PAM is starting with {} edges in graph.".format(len(g)))
    print("PAM LSS Number of Clusters Chosen: {}".format(k))

    timer.start()
    oracle_plugin = LSS(g, order_val, prime_nodes, oracle)
    p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
    timer.end()
    if verbose:
        print("LSS: ", p.centroids)
    print("{} COUNT LSS: {}\nTime(LSS): {}\n\n".format("PAM", count, timer.time_elapsed))
    print("Total LSS + Prime: {}".format(count + priming_cost))


def pam_NodeLandmark(g_prime, pr, order_val, timer, oracle, copy_centroids, k, scale_k, verbose):
    """
    Jees New code the nodeLandmark
    """
    print("Experiment Starting NodeLandMark ( pam )\n")

    import math
    global full_mat, count

    # k_list = [math.ceil(math.log(order_val)) - 2, math.ceil(math.log(order_val)), math.ceil(math.log(order_val)) + 2,
    # math.ceil(math.log(order_val)) + 4]
    v = math.ceil(math.log(order_val, 2))
    k_list = [v * scale_k, v * scale_k * 2, v * scale_k * 3, v * scale_k * 4, v * scale_k * 5, v * scale_k * 6]
    k_list = k_list[:1]
    print("Value of K: {}, Scale Factor: {}".format(k_list, scale_k))
    # oracle = flicker_oracle()
    print("PAM LAESA Number of Clusters Chosen: {}".format(k))

    for k_ in k_list:
        # g_prime = copy.copy(g)
        g = {}
        g_prime = g
        print("Number of Nodes in the start of our the LAESA: {}".format(len(g_prime)))

        centroids = copy.copy(copy_centroids)
        timer.start()
        oracle_plugin = NodeLandMarkRandom(g_prime, k_, oracle, order_val)
        p = pam_plugin(oracle, oracle_plugin, order_val, k, centroids)
        timer.end()
        if verbose:
            print("TriSearch: ", p.centroids)
        print("Node Landmark [ LAESA ]- order_val: {}".format(order_val))
        print("COUNT node Landmark {}\nk:{}\nTime: {}\n\n".format(count, k, timer.time_elapsed))


# def helper_pam_plugin(g, full_mat1, g_mat, measure, kind, order_val, k, algo, scheme_chooser_list, three=False):
def helper_pam_plugin(order_val, k, scheme_chooser_list, scale_k, oracle_chooser, three = False):
    global full_mat, count
    centroids = sample(list(range(order_val)), k)
    copy_centroids = copy.copy(centroids)

    global full_mat, count

    # Choosing the right oracle
    oracle = None
    if oracle_chooser == 1:
        oracle = flicker_oracle()
    elif oracle_chooser == 2:
        oracle = SFOracle(debug=False)
    elif oracle_chooser == 3:
        oracle = UCIUrbanOracle(debug=False)
        # oracle = SFOracle(debug=True)
    assert oracle is not None

    print(math.ceil(math.log(order_val, 2)))
    k_leaesa = math.ceil(math.log(order_val, 2)) * scale_k
    print("Value of K: {}, Scale Factor: {}".format(k_leaesa, scale_k))

    timer = Timer()
    timer.start()
    oracle_plugin = NodeLandMarkRandom({}, k_leaesa, oracle, order_val)
    timer.end()
    priming_cost = copy.copy(count)

    print("Priming Cost: {}\nTime for priming: {}\n".format(priming_cost, oracle_plugin.time2prime))



    centroids = copy.copy(copy_centroids)
    timer.start()
    # pr = vanila_pam(time_waste_oracle, order_val, k, centroids)
    pr = vanila_pam(oracle, order_val, k, centroids)

    print("Vanila: ", pr.centroids)
    # oracle, n, k, centroids = None):
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("COUNT Without Plugin ", count, timer.time_elapsed, "\n\n")

    if "SW" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        # pam_SW(pr, measure, kind, order_val, timer, algo, copy_centroids, k)
        pam_SW(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, verbose=False)

    if "DSS" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        # pam_DSS(pr, measure, kind, order_val, timer, algo, copy_centroids, k)
        pam_DSS(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, verbose=False)

    if "TSS" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        # pam_TSS(pr, measure, kind, order_val, timer, algo, copy_centroids, k)
        pam_TSS(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, verbose=False)

    if "LSS" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        # pam_LSS(pr, measure, kind, order_val, timer, algo, copy_centroids, k)
        pam_LSS(g_prime, pr, order_val, timer, oracle, copy_centroids, k, priming_cost, prime_nodes=k_leaesa, prime_edges=k_leaesa, verbose=False)

    if "NLM" in scheme_chooser_list:
        g_prime = copy.copy(oracle_plugin.G)
        print("The graph has total this many edges at start of NLM: {}".format(len(g_prime)))
        pam_NodeLandmark({}, pr, order_val, timer, oracle, copy_centroids, k, scale_k, verbose=False)


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


# def clarans_SW(pr, measure, kind, order_val, timer, algo, k):
def clarans_SW(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug):

    print("Experiment Starting SW (CLARANS)\n")

    global full_mat, count
    g = g
    # g = {}
    print_bool = True
    if len(g) == 0:
        print_bool = False

    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = clarans_plugin(order_val, no_clusters, oracle_plugin, oracle, num_local=num_local)
    timer.end()

    if debug:
        print("Plug-in(CLA): ", p.centroids)


    # sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(
    #     timer.time_elapsed - obj_sw.update_time) + "\n"
    # print("COUNT Sasha Wang: {}, Time(total): {}, Time(SP): {}\n\n".
    #         format(count, timer.time_elapsed, obj_sw.update_time))

    print("[{}] COUNT Sasha Wang: {}\norder_val:{}\nNo of Clusters: {}\nTime(total): {}, Time(SP): {}\n\n".
          format(algo_name, count, order_val, no_clusters, timer.time_elapsed, obj_sw.update_time))
    print("*" * 70)
    if print_bool:
        print("[SW][order_val: {}][#Clusters: {}] sum of ( algo + prime ) cost is: {}".
              format(order_val, no_clusters, count + prime_cost))
        print("*" * 70)
        return count + prime_cost
    else:
        print("[SW][order_val: {}][#Clusters: {}] sum of ( algo + prime[0] ) cost is: {}".
              format(order_val, no_clusters, count))
        print("*" * 70)
        return count


# def clarans_DSS(pr, measure, kind, order_val, timer, algo, k):
def clarans_DSS(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug):
    """
    DSS Solution Scheme
    """
    print("Experiment Starting DSS (CLARANS)\n")

    global full_mat, count
    g = g
    # g = {}
    print_bool = True
    if len(g) == 0:
        print_bool = False


    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = clarans_plugin(order_val, no_clusters, oracle_plugin, oracle, num_local=num_local)
    timer.end()
    if debug:
        print("DSS: ", p.centroids)
    # print("DSS Experiments - PAM\nmeasure: {}, kind: {}, order_val: {}".format(measure, kind, order_val))
    print("{} DSS COUNT {}\nTime: {}\n\n".format(algo_name, count, timer.time_elapsed))
    print("*" * 70)
    if print_bool:
        print("[DSS][order_val: {}][#Clusters: {}] sum of ( algo + prime ) cost is: {}".
              format(order_val, no_clusters, count + prime_cost))
        print("*" * 70)
        return count + prime_cost
    else:
        print("[DSS][order_val: {}][#Clusters: {}] sum of ( algo + prime[0] ) cost is: {}".
              format(order_val, no_clusters, count))
        print("*" * 70)
        return count


# def clarans_TSS(pr, measure, kind, order_val, timer, algo, k):
def clarans_TSS(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug):
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
    print("Experiment Starting TSS (CLARANS)\n")

    global full_mat, count
    g = g
    # g = {}
    print_bool = True
    if len(g) == 0:
        print_bool = False
    print("No of edges in input to TSS: {}".format(len(g)))
    timer.start()
    oracle_plugin = IntersectTriSearch(g, order_val)
    p = clarans_plugin(order_val, no_clusters, oracle_plugin, oracle, num_local=num_local)
    timer.end()

    if debug:
        print("TriSearch: ", p.centroids)
    # print(
    #     "IntersctionTriSearch Experiments - CLARANS\nmeasure: {}, kind: {}, order_val: {}".format(measure, kind,
    #                                                                                               order_val))
    print("[{}]COUNT intersect Tri-search: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".
          format(algo_name, count, timer.time_elapsed - oracle_plugin.sp_time, oracle_plugin.sp_time))

    print("*" * 70)
    if print_bool:
        print("[TSS][order_val: {}][#Clusters: {}] sum of ( algo + prime ) cost is: {}".
              format(order_val, no_clusters, count + prime_cost))
        print("*" * 70)
        return count + prime_cost
    else:
        print("[TSS][order_val: {}][#Clusters: {}] sum of ( algo + prime[0] ) cost is: {}".
              format(order_val, no_clusters, count))
        print("*" * 70)
        return count


# def clarans_LSS(pr, measure, kind, order_val, timer, algo, k):
def clarans_LSS(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug):
    """
    code to test the landmark based methods
    """
    print("Experiment Starting LSS (CLARANS)\n")

    global full_mat, count
    g = g
    g = {}

    print_bool = True
    if len(g) == 0:
        print_bool = False


    timer.start()
    oracle_plugin = LSS(g, order_val, 6, oracle)
    p = clarans_plugin(order_val, no_clusters, oracle_plugin, oracle, num_local=num_local)
    timer.end()

    if debug:
        print("LSS: ", p.centroids)

    print("{} COUNT LSS: {}\nTime(LSS): {}\n\n".format(algo_name, count, timer.time_elapsed))

    print("*" * 70)
    if print_bool:
        print("[LSS][order_val: {}][#Clusters: {}] sum of ( algo + prime ) cost is: {}".
              format(order_val, no_clusters, count + prime_cost))
        print("*" * 70)
        return count + prime_cost
    else:
        print("[LSS][order_val: {}][#Clusters: {}] sum of ( algo + prime[0] ) cost is: {}".
              format(order_val, no_clusters, count))
        print("*" * 70)
        return count


def clarans_NodeLandmark(g, order_val, timer, no_clusters, oracle, algo_name, scale_k, prime_cost, num_local, debug):
    """
    Jees New code the nodeLandmark
    """
    print("Experiment Starting NodeLandMark ({})\n".format(algo_name))

    import math
    global full_mat, count

    print_bool = True

    # k_list = [math.ceil(math.log(order_val)) - 2, math.ceil(math.log(order_val)), math.ceil(math.log(order_val)) + 2,
    # math.ceil(math.log(order_val)) + 4]
    v = math.ceil(math.log(order_val, 2))
    k_list = [v * scale_k, v * scale_k * 2, v * scale_k * 3, v * scale_k * 4, v * scale_k * 5, v * scale_k * 6]
    # k_list = k_list[:1]
    print("Value of K: {}, Scale Factor: {}".format(k_list, scale_k))
    # oracle = flicker_oracle()
    for k in k_list:
        # g_prime = copy.copy(g)
        g_prime = {}
        if len(g_prime) == 0:
            print_bool = False

        print("Number of Nodes in the start of our the LAESA: {}".format(len(g_prime)))
        timer.start()
        oracle_plugin = NodeLandMarkRandom(g_prime, k, oracle, order_val)
        p = clarans_plugin(order_val, no_clusters, oracle_plugin, oracle, num_local=num_local)
        timer.end()

        if debug:
            print("Plug-in(Nav): ", p.centroids)
        print("*" * 90)
        print("Time(LESA): {}".format(timer.time_elapsed))
        if print_bool:
            print("[LESA][order_val: {}][LandMarks: {}]  sum of ( algo + prime ) cost is: {}".
                  format(order_val, k, count + prime_cost))
            print("*" * 90)
            # return count + prime_cost
        else:
            print("[LESA][order_val: {}][LandMarks: {}]  sum of ( algo + prime[0] ) cost is: {}".
                  format(order_val, k, count))
            print("*" * 90)
            # return count


# def helper_clarans_plugin(g, full_mat1, g_mat, measure, kind, order_val, k, algo, scheme_chooser_list, three=False):
def helper_clarans_plugin(no_clusters, order_val, scheme_chooser_list, oracle_chooser, scale_k, debug=False):
    global full_mat, count

    # g, full_mat, g_mat = g, full_mat1, g_mat
    timer = Timer()

    print("\nClarans Experiments starting\n")

    # Choosing the right oracle
    oracle = None
    if oracle_chooser == 1:
        oracle = flicker_oracle()
    elif oracle_chooser == 2:
        oracle = SFOracle(debug=False)
    elif oracle_chooser == 3:
        oracle = UCIUrbanOracle(debug=False)
        # oracle = SFOracle(debug=True)
    assert oracle is not None

    timer = Timer()
    # print("Log2(Order_val): {}".format(math.ceil(math.log(order_val, 2))))
    k_landmarks = math.ceil(math.log(order_val, 2)) * scale_k
    print("Order_val: {}\nNo of Clusters: {}\nValue of Priming K: {}, Scale Factor: {}\n".
          format(order_val, no_clusters, k_landmarks, scale_k))

    timer.start()
    oracle_plugin = NodeLandMarkRandom({}, k_landmarks, oracle, order_val)
    timer.end()
    print("\nPriming Cost: {}\nTime for priming: {}\n".format(count, oracle_plugin.time2prime))
    print("\nK Landmarks: {}".format(oracle_plugin.k_landmarks))
    prime_cost = copy.copy(count)

    num_local = 3
    timer.start()
    pr = vanila_clarans(get_real_counter_oracle(oracle), order_val, no_clusters, num_local=num_local)
    timer.end()
    if debug:
        print("Plug-in(CLA): ", pr.centroids)

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("[Clarans] COUNT Without Plugin {} Time: {}\n\n".format(count, timer.time_elapsed))
    valina_count = copy.copy(count)

    tss_cnt = lss_cnt = dss_cnt = nlm_cnt = sw_cnt = 0

    algo_name = "Clarans"
    if "SW" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        sw_cnt = clarans_SW(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug)
        date_time_printer()
    if "DSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        dss_cnt = clarans_DSS(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug)
        date_time_printer()
    if "TSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        tss_cnt = clarans_TSS(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug)
        date_time_printer()
    if "LSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        lss_cnt = clarans_LSS(g, order_val, timer, no_clusters, oracle, algo_name, prime_cost, num_local, debug)
        date_time_printer()
    if "NLM" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        # nlm_cnt =
        clarans_NodeLandmark(g, order_val, timer, no_clusters, oracle, algo_name, scale_k, prime_cost, num_local, debug)
    print("$" * 90)

    print("\nSummmary - [Order Val: {}]\n".format(order_val))
    if "TSS" in scheme_chooser_list:
        print("Percentage savings Vanila[{}] Vs TSS[{}]: {}\n".format(valina_count, tss_cnt,
                                                                      ((valina_count - tss_cnt) * 100 / valina_count)))
    if "NLM" in scheme_chooser_list:
        print("Percentage savings Vanila[{}] Vs LAESA[{}]: {}\n".format(valina_count, nlm_cnt,
                                                                        ((valina_count - nlm_cnt) * 100 / valina_count)))
    # if "TSS" in scheme_chooser_list and "NLM" in scheme_chooser_list:
    #     print("Percentage savings LAESA[{}] Vs TSS[{}]: {}\n".format(nlm_cnt, tss_cnt,
    #                                                                  ((nlm_cnt - tss_cnt) * 100 / nlm_cnt)))
    print("$" * 90)


# def navarro_SW(pr, measure, kind, order_val, timer, algo, k):
def navarro_SW(g, order_val, timer, kNN, oracle, algo, prime_cost, debug):
    print("Experiment Starting SW (NAVARRO)\n")
    print_bool = True
    global full_mat, count
    g = g
    # g = {}

    if len(g) == 0:
        print_bool = False
    timer.start()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    oracle_plugin = obj_sw
    p = knnrp(order_val, kNN, oracle, oracle_plugin)
    p.knn_queries()
    if debug:
        print("SW - Plug-in(Nav): ", p.NHA)
    timer.end()

    # sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(
    #     timer.time_elapsed - obj_sw.update_time) + "\n"
    print("[{}] COUNT Sasha Wang: {}\norder_val:{}\nkNN: {}\nTime(total): {}, Time(SP): {}\n\n".
          format(algo, count, order_val, kNN, timer.time_elapsed, obj_sw.update_time))
    print("*" * 70)
    if print_bool:
        print("[SW][order_val: {}][KNN: {}] sum of ( algo + prime ) cost is: {}".format(order_val, kNN,
                                                                                        count + prime_cost))
        print("*" * 70)
        return count + prime_cost
    else:
        print("[SW][order_val: {}][KNN: {}] sum of ( algo + prime[0] ) cost is: {}".format(order_val, kNN, count))
        print("*" * 70)
        return count


# def navarro_DSS(pr, measure, kind, order_val, timer, algo, k):
def navarro_DSS(g, order_val, timer, kNN, oracle, algo, prime_cost, debug):
    """
    DSS Solution Scheme
    """
    print_bool = True
    print("Experiment Starting DSS (NAVARRO)\n")

    global full_mat, count
    g = g
    # g = {}

    if len(g) == 0:
        print_bool = False

    timer.start()
    obj_dss = DSS(g, order_val)
    oracle_plugin = obj_dss
    p = knnrp(order_val, kNN, oracle, oracle_plugin)
    p.knn_queries()
    timer.end()
    if debug:
        print("Plug-in(Nav): ", p.NHA)
    print("order_val: {}".format(order_val))
    print("[{}] DSS COUNT {}\norder_val: {}\nkNN: {}\nTime: {}\n\n".
          format(algo, count, order_val, kNN, timer.time_elapsed))
    print("*" * 70)
    if print_bool:
        print("[DSS][order_val: {}][KNN: {}]  sum of ( algo + prime ) cost is: {}".format(order_val, kNN,
                                                                                          count + prime_cost))
        print("*" * 70)
        return count + prime_cost
    else:
        print("[DSS][order_val: {}][KNN: {}]  sum of ( algo + prime[0] ) cost is: {}".format(order_val, kNN, count))
        print("*" * 70)
        return count


# def navarro_TSS(pr, measure, kind, order_val, timer, algo, k):
def navarro_TSS(g, order_val, timer, kNN, oracle, algo, prime_cost, debug):
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
    print_bool = True
    print("Experiment Starting TSS (NAVARRO)\n")

    global full_mat, count
    g = g
    # g = {}
    if len(g) == 0:
        print_bool = False

    print("Number of objects in the graph: {}".format(len(g)))
    timer.start()
    oracle_plugin = IntersectTriSearch(g, order_val)
    p = knnrp(order_val, kNN, oracle, oracle_plugin)
    p.knn_queries()
    # print("Plug-in(Nav): ", p.NHA)
    timer.end()
    if debug:
        print("Plug-in(Nav): ", p.NHA)

    print("[{}] COUNT intersect Tri-search: {}\norder_val: {}\nkNN: {}\nLB Time(Tri): {}\nUB Time(Tri): {}\n\n".
          format(algo, count, order_val, kNN, timer.time_elapsed - oracle_plugin.sp_time, oracle_plugin.sp_time))
    print("*" * 70)
    if print_bool:
        print("[TSS][order_val: {}][KNN: {}]  sum of ( algo + prime ) cost is: {}".format(order_val, kNN,
                                                                                          count + prime_cost))
        print("*" * 70)
        return count + prime_cost, p
    else:
        print("[TSS][order_val: {}][KNN: {}]  sum of ( algo + prime[0] ) cost is: {}".format(order_val, kNN, count))
        print("*" * 70)
        return count, p


# def navarro_LSS(pr, measure, kind, order_val, timer, algo, k):
def navarro_LSS(g, order_val, timer, kNN, scale_k, oracle, algo, prime_cost, debug):
    """
    code to test the landmark based methods
    """
    print_bool = True
    print("Experiment Starting LSS (NAVARRO)\n")

    global full_mat, count
    g = g
    # g = {}

    if len(g) == 0:
        print_bool = False

    v = math.ceil(math.log(order_val, 2))
    k_list = [v * scale_k, v * scale_k * 2, v * scale_k * 3, v * scale_k * 4, v * scale_k * 5, v * scale_k * 6]
    k_list = k_list[:1]

    timer.start()
    oracle_plugin = LSS(g, order_val, k_list[0], oracle)
    p = knnrp(order_val, kNN, oracle, oracle_plugin)
    p.knn_queries()
    timer.end()

    if debug:
        print("Plug-in(Nav): ", p.NHA)
    print("[{}] COUNT LSS: {}\norder_val: {}\nkNN: {}\nk_landmark: {}\nTime(LSS): {}\n\n".
          format(algo, count, order_val, kNN, k_list[0], timer.time_elapsed))
    print("*" * 70)
    if print_bool:
        print("[LSS][order_val: {}][KNN: {}]  sum of ( algo + prime ) cost is: {}".format(order_val, kNN,
                                                                                          count + prime_cost))
        print("*" * 70)
        return count + prime_cost
    else:
        print("[LSS][order_val: {}][KNN: {}]  sum of ( algo + prime[0] ) cost is: {}".format(order_val, kNN, count))
        print("*" * 70)
        return count


def navarro_NodeLandmark(g, order_val, timer, kNN, scale_k, oracle, algo, prime_cost, debug):
    """
    Jees New code the nodeLandmark
    """
    print("Experiment Starting NodeLandMark (Navarro)\n")

    import math
    global full_mat, count

    print_bool = True

    # k_list = [math.ceil(math.log(order_val)) - 2, math.ceil(math.log(order_val)), math.ceil(math.log(order_val)) + 2,
    # math.ceil(math.log(order_val)) + 4]
    v = math.ceil(math.log(order_val, 2))
    k_list = [v * scale_k, v * scale_k * 2, v * scale_k * 3, v * scale_k * 4, v * scale_k * 5, v * scale_k * 6]
    k_list = k_list[:1]
    print("Value of K: {}, Scale Factor: {}".format(k_list, scale_k))
    # oracle = flicker_oracle()
    for k in k_list:
        # g_prime = copy.copy(g)
        g_prime = {}
        if len(g_prime) == 0:
            print_bool = False

        print("Number of Nodes in the start of our the LAESA: {}".format(len(g_prime)))
        timer.start()
        oracle_plugin = NodeLandMarkRandom(g_prime, k, oracle, order_val)
        p = knnrp(order_val, kNN, oracle, oracle_plugin)
        p.knn_queries()
        timer.end()

        if debug:
            print("Plug-in(Nav): ", p.NHA)
        print("*" * 90)
        print("Time(LESA): {}".format(timer.time_elapsed))
        if print_bool:
            print("[LESA][order_val: {}][KNN: {}][LandMarks: {}]  sum of ( algo + prime ) cost is: {}".
                  format(order_val, kNN, k, count + prime_cost))
            print("*" * 90)
            return count + prime_cost, p
        else:
            print("[LESA][order_val: {}][KNN: {}][LandMarks: {}]  sum of ( algo + prime[0] ) cost is: {}".
                  format(order_val, kNN, k, count))
            print("*" * 90)
            return count, p

        # print("*" * 70)

        # print("[{}] COUNT LESA: {}\norder_val:{}\nkNN:{}\nk_lanmark: {}\nTime(LESA): {}\n\n".
        #       format(algo, count, order_val, kNN, k, timer.time_elapsed))
        # print("*" * 70)


# helper_navarro_plugin(order_val=order_val, scheme_chooser_list=scheme_chooser_list, oracle_chooser, scale_k)
def helper_navarro_plugin(kNN, order_val, scheme_chooser_list, oracle_chooser, scale_k):
    global full_mat, count
    print("\nNavarro Experiments starting\n")
    debug = False
    # debug = True

    # Choosing the right oracle
    oracle = None
    if oracle_chooser == 1:
        oracle = flicker_oracle()
    elif oracle_chooser == 2:
        oracle = SFOracle(debug=False)
    elif oracle_chooser == 3:
        oracle = UCIUrbanOracle(debug=False)
        # oracle = SFOracle(debug=True)
    assert oracle is not None

    timer = Timer()
    # print("Log2(Order_val): {}".format(math.ceil(math.log(order_val, 2))))
    k_landmarks = math.ceil(math.log(order_val, 2)) * scale_k
    print("Order_val: {}\nkNN: {}\nValue of K: {}, Scale Factor: {}\n".format(order_val, kNN, k_landmarks, scale_k))

    timer.start()
    oracle_plugin = NodeLandMarkRandom({}, k_landmarks, oracle, order_val)
    timer.end()
    print("\nPriming Cost: {}\nTime for priming: {}\n".format(count, oracle_plugin.time2prime))
    prime_cost = copy.copy(count)

    timer.start()
    pr = vanila_knnrp(order_val, kNN, oracle)
    pr.knn_queries()
    if debug:
        print("Plug-in(Nav): ", pr.NHA)
    timer.end()

    base_algo_results = "COUNT Without Plugin " + str(count) + " Time " + str(timer.time_elapsed) + "\n"
    print("COUNT Without Plugin ", count, timer.time_elapsed, "\n\n")
    valina_count = copy.copy(count)

    sw_cnt = tss_cnt = dss_cnt = lss_cnt = nlm_cnt = 0

    if "SW" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        sw_cnt = navarro_SW(g, order_val, timer, kNN, oracle, "Navarro", prime_cost, debug)
        # navarro_SW(pr, measure, kind, order_val, timer, algo, k)
        date_time_printer()

    if "DSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        dss_cnt = navarro_DSS(g, order_val, timer, kNN, oracle, "Navarro", prime_cost, debug)
        # navarro_DSS(pr, measure, kind, order_val, timer, algo, k)
        date_time_printer()

    if "TSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        tss_cnt, p_tss = navarro_TSS(g, order_val, timer, kNN, oracle, "Navarro", prime_cost, debug)
        date_time_printer()

    if "LSS" in scheme_chooser_list:
        g = copy.copy(oracle_plugin.G)
        # g = {}
        # navarro_LSS(pr, measure, kind, order_val, timer, algo, k)
        lss_cnt = navarro_LSS(g, order_val, timer, kNN, scale_k, oracle, "Navarro", prime_cost, debug)
        date_time_printer()

    if "NLM" in scheme_chooser_list:
        # g = copy.copy(oracle_plugin.G)
        g = {}
        nlm_cnt, p_nlm = navarro_NodeLandmark(g, order_val, timer, kNN, scale_k, oracle, "Navarro", prime_cost, debug)
        date_time_printer()

    print("$" * 90)

    print("\nSummmary - [Order Val: {}][kNN: {}]\n".format(order_val, kNN))
    if "TSS" in scheme_chooser_list:
        print("Percentage savings Vanila[{}] Vs TSS[{}]: {}\n".format(valina_count, tss_cnt,
                                                                      ((valina_count - tss_cnt) * 100 / valina_count)))
        if pr.NHA == p_tss.NHA:
            print("Both are Sme in pr.NHA and p_tss.NHA")
    if "NLM" in scheme_chooser_list:
        print("Percentage savings Vanila[{}] Vs LAESA[{}]: {}\n".format(valina_count, nlm_cnt,
                                                                        ((
                                                                                     valina_count - nlm_cnt) * 100 / valina_count)))
        if pr.NHA == p_nlm.NHA:
            print("Both are Sme in pr.NHA and p_nlm.NHA")
    if "TSS" in scheme_chooser_list and "NLM" in scheme_chooser_list:
        print("Percentage savings LAESA[{}] Vs TSS[{}]: {}\n".format(nlm_cnt, tss_cnt,
                                                                     ((nlm_cnt - tss_cnt) * 100 / nlm_cnt)))
    print("$" * 90)
