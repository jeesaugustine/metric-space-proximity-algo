import sys
import os
from statistics import mean


def read_process(file_name):
    lines = None
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # lines = [i.strip()]
    return [float(i.strip()) for i in lines]


def get_pair_wise_errors_lb_ub(lb1, ub1, lb2, ub2):
    i = 0
    j = 0
    assert len(lb1) == len(ub1) == len(lb2) == len(ub2)
    l_error = 0
    u_error = 0
    l_error_max = 0
    l_error_min = 100000
    u_error_max = 0
    u_error_min = 100000
    for k in range(len(lb1)):
        if not lb1[k] == lb2[k]:
            tmp = abs(lb1[k] - lb2[k])/max(lb1[k], lb2[k])
            l_error_max = max(l_error_max, tmp)
            l_error_min = min(l_error_min, tmp)
            l_error += tmp
            i += 1
        if not ub1[k] == ub2[k]:
            temp = abs(ub1[k] - ub2[k])/max(ub1[k], ub2[k])
            u_error += temp
            u_error_max = max(u_error_max, temp)
            u_error_min = min(u_error_min, temp)
            j += 1

    if i > 0:
        print("Found Difference in Bounds: LB")
        l_error = l_error / i
    if j > 0:
        print("Found Difference in Bounds: UB")
        u_error = u_error / j
    return l_error, u_error, mean(lb2), mean(ub2), l_error_max, l_error_min, u_error_max, u_error_min


def plotting_master(order_val):
    # import pdb
    # pdb.set_trace()
    list_algos = ['sw', 'dss', 'tss', 'laesa', 'tlaesa']
    lbs = {}
    ubs = {}
    for algo in list_algos:
        lb_path = os.path.join(os.getcwd(), "bounds_compare_results", "bounds_{}_{}".format(order_val, algo),
                               "lower_bounds_{}_{}.lb".format(order_val, algo))
        ub_path = os.path.join(os.getcwd(), "bounds_compare_results", "bounds_{}_{}".format(order_val, algo),
                               "upper_bounds_{}_{}.ub".format(order_val, algo))

        lbs[algo] = read_process(lb_path)
        ubs[algo] = read_process(ub_path)
    base_avg_lb = mean(lbs['sw'])
    base_avg_ub = mean(ubs['sw'])
    errors_lb_dss, errors_ub_dss, avg_lb_dss, avg_ub_dss, \
    l_error_max_dss, l_error_min_dss, u_error_max_dss, u_error_min_dss = get_pair_wise_errors_lb_ub(lbs['sw'], ubs['sw'], lbs['dss'], ubs['dss'])

    errors_lb_tss, errors_ub_tss, avg_lb_tss, avg_ub_tss,\
    l_error_max_tss, l_error_min_tss, u_error_max_tss, u_error_min_tss = get_pair_wise_errors_lb_ub(lbs['sw'], ubs['sw'], lbs['tss'], ubs['tss'])

    errors_lb_laesa, errors_ub_laesa, avg_lb_laesa, avg_ub_laesa, \
    l_error_max_laesa, l_error_min_laesa, u_error_max_laesa, u_error_min_laesa = get_pair_wise_errors_lb_ub(lbs['sw'], ubs['sw'], lbs['laesa'], ubs['laesa'])

    errors_lb_tlaesa, errors_ub_tlaesa, avg_lb_tlaesa, avg_ub_tlaesa, \
    l_error_max_tlaesa, l_error_min_tlaesa, u_error_max_tlaesa, u_error_min_tlaesa = get_pair_wise_errors_lb_ub(lbs['sw'], ubs['sw'], lbs['tlaesa'], ubs['tlaesa'])

    print("average ADM bound ( lower ): {}\naverage ADM bound ( upper ): {}".format(base_avg_lb, base_avg_ub))
    print("average Error DSS ( lower ): {}\naverage Error DSS ( upper ): {}\navg DSS lb: {}\navg DSS ub: {}".
          format(errors_lb_dss, errors_ub_dss, avg_lb_dss, avg_ub_dss))
    print("average Error TSS ( lower ): {}\naverage Error TSS ( upper ): {}\navg TSS lb: {}\navg TSS ub: {}".
          format(errors_lb_tss, errors_ub_tss, avg_lb_tss, avg_ub_tss))
    print("average Error LAESA ( lower ): {}\naverage Error LAESA ( upper ): {}\navg LAESA lb: {}\navg LAESA ub: {}".
          format(errors_lb_laesa, errors_ub_laesa,avg_lb_laesa, avg_ub_laesa))
    print("average Error TLAESA ( lower ): {}\naverage Error TLAESA ( upper ): {}\navg TLAESA lb: {}\navg TLAESA ub: {}".
          format(errors_lb_tlaesa, errors_ub_tlaesa,avg_lb_tlaesa, avg_ub_tlaesa))

    print(base_avg_lb, base_avg_ub, avg_lb_dss, avg_ub_dss, avg_lb_tss, avg_ub_tss, avg_lb_laesa, avg_ub_laesa, avg_lb_tlaesa, avg_ub_tlaesa)
    print(0, 0, l_error_max_dss, u_error_max_dss, l_error_max_tss, u_error_max_tss, l_error_max_laesa, u_error_max_laesa, l_error_max_tlaesa, u_error_max_tlaesa)
    print(0, 0, l_error_min_dss, u_error_min_dss, l_error_min_tss, u_error_min_tss, l_error_min_laesa, u_error_min_laesa, l_error_min_tlaesa, u_error_min_tlaesa)

    print("alternate")
    print(base_avg_lb, avg_lb_dss, avg_lb_tss, avg_lb_laesa, avg_lb_tlaesa, base_avg_ub, avg_ub_dss, avg_ub_tss, avg_ub_laesa, avg_ub_tlaesa)
    print(0, l_error_max_dss, l_error_max_tss, l_error_max_laesa, l_error_max_tlaesa, 0, u_error_max_dss, u_error_max_tss,
          u_error_max_laesa, u_error_max_tlaesa)
    print(0, l_error_min_dss, l_error_min_tss, l_error_min_laesa, l_error_min_tlaesa, 0, u_error_min_dss, u_error_min_tss,
          u_error_min_laesa, u_error_min_tlaesa)


if __name__ == "__main__":
    order_val = int(sys.argv[1])
    plotting_master(order_val)
