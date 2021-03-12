import sys

# comment the line later 
# from helper_paper_experiments_scale import *
from tlaesa_helper_paper_experiments_scale import *

if __name__ == "__main__":
    distance_measure = int(sys.argv[1])
    kind = int(sys.argv[2])
    order_val = int(sys.argv[3])
    algo_key = int(sys.argv[4])
    scale_k = int(sys.argv[5])
    oracle_chooser = int(sys.argv[6])

    print("algo Key: {}\nOrder Val: {}".format(algo_key, order_val))
    # scheme_chooser_list = ["DSS", "TSS", "LSS", "NLM", "LBT", "LBUB"]
    # scheme_chooser_list = ["TSS", "NLM"]
    scheme_chooser_list = ["NLM", "TLAESA", "TSS"]
    # scheme_chooser_list = ["TSS", "NLM"]
    # scheme_chooser_list = ["LSS"]
    # scheme_chooser_list = ["DSS", "TSS", "SW", "NLM"]

    if algo_key == 0:
        helper_prims_plugin(order_val, scheme_chooser_list, scale_k, oracle_chooser)
    if algo_key == 1:
        helper_kruskals_plugin(order_val, scheme_chooser_list, scale_k, oracle_chooser, three=False)
        # helper_kruskals_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, key_list[algo_key], scheme_chooser_list)
    if algo_key == 2:
        # for k in [2, 5, 10, 20]:
        for k in [2, 5, 10, 20]:
            helper_pam_plugin(order_val, k, scheme_chooser_list, scale_k, oracle_chooser, three=False)
            # helper_pam_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, k, key_list[algo_key],
            #                   scheme_chooser_list)
            print("*$*"*20, "\n")
    if algo_key == 3:
        for no_centers in [2, 5, 10, 20]:
            helper_clarans_plugin(no_centers, order_val, scheme_chooser_list=scheme_chooser_list,
                                  oracle_chooser=oracle_chooser, scale_k=scale_k)
        # for k in [2, 5, 10, 20]:
        #     helper_clarans_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, k, key_list[algo_key],
        #                           scheme_chooser_list)
            # print("*$*" * 20, "\n")
    if algo_key == 4:
        # for k in [2, 5, 10, 20]:
        for kNN in [5, 10, 20, 50]:
            helper_navarro_plugin(kNN, order_val=order_val, scheme_chooser_list=scheme_chooser_list,
                                  oracle_chooser=oracle_chooser, scale_k=scale_k)
            # helper_navarro_plugin(g, full_mat, g_mat, distance_measure, kind, order_val, k, key_list[algo_key], scheme_chooser_list)
    #         print("*$*" * 20, "\n")
    #
    # helper_pam_test(distance_measure, kind, order_val, 5)
