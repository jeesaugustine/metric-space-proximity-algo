from edge_landmark_sampling import EdgeLandMark
from sasha_wang import SashaWang

import pickle
# import networkx as nx
# from helper import _get_matrix_from_adj_dict, _bfs
import time

import os
import sys
sys.path.append('..')


def file_writer(verbose, g_name, n, landmarks, pretext, print_statement):
    if verbose:
        print(print_statement)
        with open(
                os.path.join("elm-results",
                             str(pretext) + g_name.split('.')[0].strip() + "_n-" + str(n) + "_k-" + str(landmarks) +
                             '.txt'), "a+") as text_file:
            text_file.write(print_statement + "=" * 50 + "\n")


def helper_lm_sampling(measure, kind, order_val, n, landmarks, verbose=True, sampling=True, sw=False):
    print("Values for the run - order_val:{}, Samples:{}, landmarks:{}".format(order_val, n, landmarks))
    distance_measure = ['normal', 'uniform', 'zipf', 'data_flicker', 'data_sf', 'data_20']
    generation_algorithms = ['Geometric', 'Renyi Erdos', 'ForrestFire', 'Barabasi']
    """g_name usually looks like this -> "normal_distances_Barabasi_64.pkl" 
    <distribution>_distances_<type-of-graph>_<#-of-nodes>.pkl """
    g_name = distance_measure[measure] + '_distances_' + generation_algorithms[kind] + '_' + str(order_val) + '.pkl'
    print('Path to the Input Graph File: {}'.format(os.path.join("/Users/jeesaugustine/git_it/distance_opti_plug_in/",
                                                                 "igraph", g_name)))
    graph = pickle.load(open(os.path.join("..", "igraph", g_name), 'rb'))

    # graph = {(0, 1): 0.3, (1, 2): 0.5, (2, 3): 0.5, (2, 4): 0.4, (4, 5): 0.6, (5, 6): 0.3}

    elm = EdgeLandMark(graph, n, order_val, Sampling=sampling)
    start = time.time()
    elm.find_paths()
    new_total_1 = elm.greedy_sampling(landmarks)
    print_string = "Sampling: True\nNo of Nodes in Graph: {} \nTotal Known Edges in Graph: {} \nTotal Right Samples(" \
                   "n): {} \nTotal Left Edges(k): {}\n Sum Lower Bounds: {}\n Time Taken:{}\n".format(
                    order_val, len(graph), n, landmarks, new_total_1, (time.time()-start)/60)
    file_writer(verbose, g_name, n=n, landmarks=landmarks, pretext="ELM_", print_statement=print_string)
        # print(elm.greedyK)

    # elm = EdgeLandMark(graph, n, order_val, Sampling=False)
    # elm.find_paths()
    # elm.greedy_sampling(k)
    # if verbose:
    #     print("Sampling: False\nNo of Nodes in Graph: {} \nTotal Known Edges in Graph: "
    #           "{} \n ".format(order_val, len(graph)))
    #     print(elm.greedyK)

    if sw:
        obj_sw = SashaWang()
        start_sw = time.time()
        obj_sw.store(graph, order_val)
        print_statement_sw = "Time taken for SW on the graph is {}\n".format((time.time() - start_sw) / 60)
        file_writer(verbose, g_name, n=n, landmarks=landmarks, pretext="SW_", print_statement=print_statement_sw)
        new_total_1 = 0
        lb_total = 0
        for i in range(order_val):
            for j in range(i + 1, order_val):
                if obj_sw.matrix[i][j] != -1:
                    new_total_1 += obj_sw.matrix[i][j]
                else:
                    lb_total += obj_sw.lb_matrix[i][j]
        print("\nThe SW LB Sum(Given): {}, The LB Total: {}".format(new_total_1, lb_total))

    # return elm.greedyK
