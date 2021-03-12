from sasha_wang import SashaWang
from helper import *
from helper import _get_matrix_from_adj_dict
from parametrized_path_search import ParamTriSearch
from helper_other_algos import *
from helper_plugins import *
from helper_paper_experiments import *
import sys, time


def read_graph(measure, kind, order_val, algo_exec):
    start = time.time()
    distance_measure = ['normal', 'uniform', 'zipf', 'data_flicker', 'data_sf', 'data_20']
    generation_algorithms = ['Geometric', 'Renyi Erdos', 'ForrestFire', 'Barabasi']
    full_mat_name = distance_measure[measure] + str(order_val) + '.pkl'
    g_name = distance_measure[measure] + '_distances_' + generation_algorithms[kind] + '_' + str(order_val) + '.pkl'
    g = pickle.load(open(os.path.join("igraph", g_name), 'rb'))
    full_mat = pickle.load(open(os.path.join("igraph", full_mat_name), 'rb'))
    g_mat = _get_matrix_from_adj_dict(g, order_val)
    print("Graph load Time: {}".format(time.time() - start))
    print("\nAlgorithm Under Test: {} [measure:{}({}), Gen Meth: {}({}), order_val:{}]\n\n".
          format(algo_exec, distance_measure[measure], measure, generation_algorithms[kind], kind, order_val))
    return g, full_mat, g_mat


if __name__ == "__main__":
    distance_measure = int(sys.argv[1])
    kind = int(sys.argv[2])
    order_val = int(sys.argv[3])
    algo_key = int(sys.argv[4])

    key_list = ['PRIMS', 'KRUSKALS', 'PAM', 'CLARANS', 'NAVARRO']
    g, full_mat, g_mat = read_graph(distance_measure, kind, order_val, algo_exec=key_list[algo_key])
    # scheme_chooser_list = ["DSS", "TSS", "LSS", "NLM", "LBT", "LBUB"]
    scheme_chooser_list = ["LSS"]
    # scheme_chooser_list = ["DSS", "TSS", "LSS", "SW"]
    if algo_key == 0:
        helper_prims_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, key_list[algo_key], scheme_chooser_list)
    if algo_key == 1:
        helper_kruskals_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, key_list[algo_key], scheme_chooser_list)
    if algo_key == 2:
        for k in [2, 5, 10, 20]:
            helper_pam_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, k, key_list[algo_key], scheme_chooser_list)
            print("*$*"*20, "\n")
    if algo_key == 3:
        for k in [2, 5, 10, 20]:
            helper_clarans_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, k, key_list[algo_key], scheme_chooser_list)
            print("*$*" * 20, "\n")
    if algo_key == 4:
        for k in [2, 5, 10, 20]:
            helper_navarro_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, k, key_list[algo_key], scheme_chooser_list)
            print("*$*" * 20, "\n")
    #
    # helper_pam_test(distance_measure, kind, order_val, 5)
