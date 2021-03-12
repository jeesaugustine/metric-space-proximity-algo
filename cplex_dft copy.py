from itertools import combinations
import datetime
import numpy as np
from datetime import datetime
import os
import time
import sys

from algorithms_with_plug_in.prims_with_plug_in import Prims as prims_plugin
from sasha_wang import SashaWang
from unified_graph_lb_ub import unified_graph_lb_ub
from LAESA import NodeLandMarkRandom
from vanila_algorithms.prims import Prims as vanila_prims
from tlaesa_helper_paper_experiments_scale import prims_SW

from dataprep_ficker1M.cure_1M_to_32K import get_oracle
from SF_POI.SF_Oracle import get_SF_ORACLE
import copy
import pdb

from docplex.mp.model import Model
# import docplex

from docplex.util.status import JobSolveStatus

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


def TestOracle(debug=False):
    print("TLAESA Testing Oracle")
    np.random.seed(20)
    dist = np.random.rand(10, 2)
    from scipy.spatial import distance
    o = lambda x, y: distance.minkowski(dist[x, :], dist[y, :], 2)

    def oracle(u, v):
        update_count()
        # print("u: {}, v: {} - dist ({})".format(u, v, o(u, v)))
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


def prims_SW(g, pr, order_val, timer, oracle, out_name, name_oracle, bounder=False):
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
            for j in range(i + 1, order_val):
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
    with open(out_name, 'a+') as f:
        f.write("Order Val: {}\t Oracle: {}\n".format(order_val, name_oracle))
        f.write("COUNT Sasha Wang: {}, Time(total): {}, Time(SP): {}\n\n".format(count, timer.time_elapsed,
                                                                                 obj_sw.update_time))


class DirFeasibilityTest:
    def __init__(self, order_val):
        self.order_val = order_val
        self.model = Model("LP model")
        self.vars = []
        self.calculated = {}
        for i in range(self.order_val-1):
            self.calculated[i] = set()
            self.vars.append([])
            for j in range(i+1, self.order_val):
                self.vars[i].append(self.model.continuous_var(name='x_'+str(i)+"_"+str(j), lb=0., ub=1.))
        # self.triangles = list(combinations(range(self.order_val), 3))
        self.triangle_constraints = {}
        for i in range(self.order_val):
            if i not in self.triangle_constraints:
                self.triangle_constraints[i] = {}
            for j in range(i+1, self.order_val):
                if j not in self.triangle_constraints[i]:
                    self.triangle_constraints[i][j] = {}
                for k in range(j+1, self.order_val):
                    self.triangle_constraints[i][j][k] = []
                    self.triangle_constraints[i][j][k].append(self.model.add_constraint(
                        self.get_var((i, j)) - self.get_var((j, k)) - self.get_var((i, k)) <=0,
                        '{}_{}_{}'.format(i, j, k))
                    )
                    self.triangle_constraints[i][j][k].append(self.model.add_constraint(
                        self.get_var((j, k)) - self.get_var((i, k)) - self.get_var((i, j))  <= 0,
                        '{}_{}_{}'.format(j, k, i))
                    )
                    self.triangle_constraints[i][j][k].append(self.model.add_constraint(
                        self.get_var((i, k)) - self.get_var((i, j)) - self.get_var((j, k)) <= 0,
                        '{}_{}_{}'.format(k, i, j))
                    )

    def get_var(self, edge):
        x, y = min(edge), max(edge)
        return self.vars[x][y-x-1]

    def is_uncalculated(self, x, y):
        u, v = min(x, y), max(x, y)
        return not (u in self.calculated and v in self.calculated[u])

    def eqn_maker(self, triples):
        eqn = None
        for t in triples:
            if eqn is None:
                eqn = t[2] * self.get_var((t[0], t[1]))
            else:
                eqn += t[2] * self.get_var((t[0], t[1]))
        return eqn

    def update(self, edge, val):
        x, y = min(edge), max(edge)
        self.calculated[x].add(y)
        self.model.add_constraint(self.get_var(edge) == val, "eq_{}_{}".format(x, y))
        for i in range(self.order_val):
            if i == x or i == y:
                continue
            if not self.is_uncalculated(i, x) and not self.is_uncalculated(i, y):
                self.delete_inequality(i, x, y)

    def solve_inequality(self, c):
        self.model.minimize(c)
        self.model.solve()
        if self.model.get_solve_status() == JobSolveStatus.INFEASIBLE_SOLUTION:
            pdb.set_trace()
        elif self.model.get_solve_status() != JobSolveStatus.OPTIMAL_SOLUTION:
            pdb.set_trace()
        else:
            # Optimal solution
            return self.model.objective_value


    def delete_inequality(self, a, b, c):
        triangle = [a, b, c]
        triangle.sort()
        if a in self.triangle_constraints and b in self.triangle_constraints[a] and c in self.triangle_constraints[a][b]:
            for cons in self.triangle_constraints[a][b][c]:
                self.model.remove_constraint(cons)
            del self.triangle_constraints[a][b][c]
            if len(self.triangle_constraints[a][b].keys()) == 0:
                del self.triangle_constraints[a][b]
                if len(self.triangle_constraints[a].keys()) == 0:
                    del self.triangle_constraints[a]


class PrimsDFT:
    def __init__(self, order, oracle, plug_in_oracle):
        # lb, ub = plug_in_oracle(u, v)
        # actual_distance = oracle(u, v)
        self.oracle = oracle
        self.plug_in_oracle = plug_in_oracle
        self.order = order
        self.mst_dict = dict()
        self.mst_path_length = None

    def mst(self, source, nodes=None):
        if nodes is None:
            nodes = list(range(self.order))
        mst_dict = dict()
        set_nodes = set(nodes)
        set_nodes.remove(source)
        removed = set([source])
        # (distance, cur_node, parent)
        heap = [[0, source, source]]
        ptr = {source: 0}
        total = 0
        self.dist_dict = []
        for i in range(self.order - 1):
            if i % 1000 == 0:
                print("Number of Edges found by primes: {}".format(i))
            candidate_set = []
            for r in removed:
                for n in set_nodes:
                    candidate_set.append((r, n))
            for r in removed:
                for n in set_nodes:
                    i = 0
                    add_edge = True
                    while i < len(candidate_set):
                        if min(r, n) == min(candidate_set[i]) and max(r, n) == max(candidate_set[i]):
                            i += 1
                            continue
                        soln_max = self.plug_in_oracle.solve_inequality(
                            self.plug_in_oracle.eqn_maker([(r, n, -1), (candidate_set[i][0], candidate_set[i][1], 1)]))
                        if -soln_max <= 0:
                            del candidate_set[i]
                        else:
                            i += 1
            min_dist = 1
            min_edge = None
            print(candidate_set)
            lookedup = []
            import pdb
            while len(candidate_set) > 0:
                dominated = False
                c = candidate_set.pop()
                x, y = min(c), max(c)
                calc = self.plug_in_oracle.is_uncalculated(x, y)
                if not calc:
                    soln_opt = self.plug_in_oracle.eqn_maker([(x, y, 1)])
                    soln_lp = self.plug_in_oracle.solve_inequality(soln_opt)
                    dist = soln_lp
                    if dist < min_dist:
                        min_dist = dist
                        min_edge = c
                    continue
                if len(lookedup) != 0:
                    comaprision_set = lookedup + candidate_set
                    for lkd_up in comaprision_set:
                        soln_max = self.plug_in_oracle.solve_inequality(
                            self.plug_in_oracle.eqn_maker([(lkd_up[0], lkd_up[1], -1), (x, y, 1)]))
                        if -soln_max <= 0:
                            print("An element got dominated {}-{}".format(x, y))
                            dominated = True
                            break
                if not dominated:
                    dist = self.oracle(x, y)
                    print("{}-{}:{}".format(x, y, dist))
                    self.plug_in_oracle.update((x, y), dist)
                    lookedup.append(c)
                    if dist < min_dist:
                        min_dist = dist
                        min_edge = (x, y)
            remove_edge = None
            if min_edge[0] in removed:
                remove_edge = min_edge[1]
            else:
                remove_edge = min_edge[0]
            removed.add(remove_edge)
            set_nodes.remove(remove_edge)
            total += min_dist
            mst_dict[min_edge] = min_dist
        # print("PRIMS: ", total)
        self.mst_dict = mst_dict
        self.mst_path_length = total
        print(self.plug_in_oracle.calculated)


def prims_DFT(order_val, oracle, timer, out_name, name_oracle):
    print("Experiment Starting Sasha Wang (Prims)\n")

    global full_mat, count
    # g = {}

    timer.start()
    oracle_plugin = DirFeasibilityTest(order_val)

    # oracle = flicker_oracle()
    p = PrimsDFT(order_val, oracle, oracle_plugin)
    p.mst(2)
    timer.end()
    print(p.mst_dict)

    print("path length from DFT: {}".format(p.mst_path_length))
    # assert abs(p.mst_path_length - pr.mst_path_length) < 0.000001
    # print("Plugin with Sasha Wang Experiments\nActual(SW) Prims Path Length: {}\nSasha Wang Prims Path Length: {}\n"
    #       "order_val: {}".
    #       format(p.mst_path_length, pr.mst_path_length, order_val))
    # sasha_wang_results = "COUNT Sasha Wang " + str(count) + " Time " + str(
    #     timer.time_elapsed - obj_sw.update_time) + "\n"
    print(
        "COUNT DFT: {}, Time(total): {}, order_val: {} \n\n".format(count, timer.time_elapsed, order_val))

    with open(out_name, 'a+') as f:
        f.write("Order Val: {}\t Oracle: {}\n".format(order_val, name_oracle))
        f.write("COUNT DFT: {}, Time(total): {}, order_val: {} \n\n".format(count, timer.time_elapsed, order_val))


def helper_prims_plugin(out_name, order_val, scheme_chooser_list=1, scale_k=1, oracle_chooser=2, bounder=False, three=False):
    global count

    oracle = None
    name_oracle = ""
    if oracle_chooser == 1:
        oracle = flicker_oracle()
        name_oracle = "Flicker Results Below.\n"
    elif oracle_chooser == 2:
        oracle = SFOracle(debug=False)
        name_oracle = "SFOracle Results Below.\n"
    elif oracle_chooser == 3:
        oracle = UCIUrbanOracle(debug=False)
        name_oracle = "UCIUrbanOracle Results Below.\n"
    elif oracle_chooser == 4:
        oracle = TestOracle()
        name_oracle = "TestOracle Results Below.\n"
        # oracle = SFOracle(debug=True)
    assert oracle is not None
    with open(out_name, 'a+') as f:
        f.write("-" * 30)
        f.write(name_oracle)
    # oracle_plugin = NodeLandMarkRandom({}, k, oracle, order_val)
    # print("Priming Cost: {}\nTime for priming: {}\n".format(count, oracle_plugin.time2prime))

    # pr = vanila_prims(order_val, time_waste_oracle)
    pr = vanila_prims(order_val, oracle)
    timer = Timer()
    timer.start()
    pr.mst(2)
    timer.end()
    base_algo_results = "COUNT Without Plugin(Vanila) " + str(count) + "\nTime(Vanila) " + str(
        timer.time_elapsed) + "\n"
    print(base_algo_results, "\n\n")

    prims_SW({}, pr, order_val, timer, oracle, out_name, name_oracle, bounder=bounder)
    date_time_printer()

    new = prims_DFT(order_val, oracle, timer, out_name, name_oracle)


file_name = "cplex_dft_scale.res"
order_val = int(sys.argv[1])
oracle_chooser = int(sys.argv[2])
helper_prims_plugin(file_name, order_val=order_val, scheme_chooser_list=1, scale_k=1, oracle_chooser=oracle_chooser,bounder=False,
                    three=False)

with open(file_name, 'a+') as f:
    f.write('*' * 20)
    f.write('\n')
