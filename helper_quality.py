from algorithms_with_plug_in.prims_with_plug_in import Prims as prims_plugin
from algorithms_with_plug_in.kruskal_with_plug_in import Kruskals as kruskals_plugin
from unified_graph_lb_ub import unified_graph_lb_ub
from sasha_wang import SashaWang
from graph_maker import NlogNGraphMaker
from helper import _get_matrix_from_adj_dict, _bfs
from algorithms_with_plug_in.pam_with_plug_in import PAM as pam_plugin
from algorithms_with_plug_in.clarans_with_plug_in import CLARANS as clarans_plugin
from random import choices, seed
from parametrized_path_search import ParamTriSearch


from vanila_algorithms.prims import Prims as vanila_prims
from vanila_algorithms.kruskals import Kruskals as vanila_kruskals
from vanila_algorithms.pam import PAM as vanila_pam
from vanila_algorithms.clarans import CLARANS as vanila_clarans

import numpy as np
import time
import pickle
import os
import pdb

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


def helper_tester(order_val, measure, kind):
    global full_mat, count
    distance_measure = ['normal', 'uniform', 'zipf']
    generation_algorithms = ['Geometric', 'Renyi Erdos', 'ForrestFire', 'Barabasi']
    full_mat_name = distance_measure[measure] + str(order_val) + '.pkl'
    g_name = distance_measure[measure] + '_distances_' + generation_algorithms[kind] + '_' + str(order_val) + '.pkl'
    g = pickle.load(open(os.path.join("igraph", g_name), 'rb'))
    full_mat = pickle.load(open(os.path.join("igraph", full_mat_name), 'rb'))
    g_mat = _get_matrix_from_adj_dict(g, order_val)

    count = 0
    obj_sw = SashaWang()
    start = time.time()
    obj_sw.store(g, order_val)
    sw_time = time.time() - start
    obj = unified_graph_lb_ub()
    start = time.time()
    obj.store(g, order_val)
    our_time = time.time() - start

    start = time.time()
    obj_tri = ParamTriSearch(2, None)
    obj_tri.store(g, order_val)
    tri_time = time.time() - start


    average_lbt = np.average(np.array(obj_sw.lb_matrix) - np.array(obj.lb_matrix))
    mean_sw = np.average(np.array(obj_sw.lb_matrix))
    mean_ours = np.average(np.array(obj.lb_matrix))
    mean_tri = np.average(np.array(obj_tri.lb_matrix))
    average_original = np.average(full_mat - np.array(obj_sw.lb_matrix))

    results_file_name = "Error_" + str(order_val) + "_" \
                        + generation_algorithms[kind] + "_" \
                        + distance_measure[measure] + "_" + ".txt"
    lbt_results = " Time SW: " \
                  + str(sw_time) + "\n"\
                  + " Time lbub: " + str(our_time) + "\n"\
                  + " Average Error with LBT " + str(average_lbt) + "\n" \
                  + " Average Error with Original " + str(average_original) \
                  + " Mean SW " + str(mean_sw) \
                  + " Mean SW " + str(mean_ours) \
                  + " Mean TriSearch " + str(mean_tri) \
                  + " Time Tri " + str(tri_time) \
                  + "\n"
    print(lbt_results)
    f = open(os.path.join("quality", results_file_name), "w+")
    f.write(lbt_results)
    f.write("\n")
    f.close()
