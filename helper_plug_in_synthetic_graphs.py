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
import numpy as np
import time
import pickle
import os

import copy


debug = False
# order_val = 100
order_val = None
# graph_maker = NlogNGraphMaker(order_val)
# g = graph_maker.get_nlogn_edges()
g = None
# full_mat = graph_maker.out
full_mat = None
g_mat = None

count = 0


def oracle(i, j):
    global full_mat, count
    count += 1
    return full_mat[i][j]


def helper_sasha_wang_saver(id_graph_to_run, type_of_graph_id):
    global g, g_mat, order_val, full_mat, count
    graph = dict()
    graph[0] = ['normal_distances_Geometric_512.pkl', 'normal_distances_Renyi Erdos_512.pkl',
                       'normal_distances_ForrestFire_512.pkl', 'normal_distances_Barabasi_512.pkl']
    graph[1] = ['uniform_distances_Geometric_512.pkl', 'uniform_distances_Renyi Erdos_512.pkl',
                        'uniform_distances_ForrestFire_512.pkl', 'uniform_distances_Barabasi_512.pkl']
    graph[2] = ['zipf_distances_Geometric_512.pkl', 'zipf_distances_Renyi Erdos_512.pkl',
                        'zipf_distances_ForrestFire_512.pkl', 'zipf_distances_Barabasi_512.pkl']

    outs = ['normal512.pkl', 'uniform512.pkl', 'zipf512.pkl']

    full_mat = pickle.load(open(os.path.join('igraph', outs[id_graph_to_run]), 'rb'))
    order_val = full_mat.shape[0]

    graph = graph[id_graph_to_run][type_of_graph_id]
    print(graph)

    ub_out_name = os.path.join("LB_UB",
                               '_'.join(['ub_sw', str(graph.split('_')[0]), graph.split('_')[2], graph.split('_')[3]]))
    lb_out_name = os.path.join("LB_UB",
                               '_'.join(['lb_sw', str(graph.split('_')[0]), graph.split('_')[2], graph.split('_')[3]]))
    g = pickle.load(open(os.path.join('igraph', graph), 'rb'))
    count = 0
    print("Graph Chosen; SW Starting.")
    start = time.time()
    obj_sw = SashaWang()
    obj_sw.store(g, order_val)
    end = time.time()
    print("Time for SW for algorithm(", '_'.join([str(graph.split('_')[0]),
                                                   graph.split('_')[2], graph.split('_')[3]]), "): ", (end-start))
    pickle.dump(np.array(obj_sw.ub_matrix), open(ub_out_name, 'wb'))
    pickle.dump(np.array(obj_sw.lb_matrix), open(lb_out_name, 'wb'))
