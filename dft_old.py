from scipy.optimize import linprog
from itertools import combinations
import datetime
import numpy as np
from datetime import datetime
import os
import time

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


class DirFeasibilityTest:
    def __init__(self, order_val):
        self.order_val = order_val
        self.triangles = list(combinations(range(self.order_val), 3))
        self.vars = dict()
        self.A_inequality = []
        self.A_equality = []
        self.b_equality = []
        self.no_vars = int((self.order_val*(self.order_val - 1))/2)
        self.A_inequality_dict = {}
        for t in self.triangles:
            t_ = list(t)
            t_.sort()
            self.A_inequality_dict["{}_{}_{}".format(t[0], t[1], t[2])] = []
            for i in range(3):
                n = [-1, -1, -1]
                n[i] = 1
                l = [0] * self.no_vars
                l[self._calc_offset((t[0], t[1]))] = n[0]
                l[self._calc_offset((t[0], t[2]))] = n[1]
                l[self._calc_offset((t[1], t[2]))] = n[2]
                self.A_inequality.append(l)
                self.A_inequality_dict["{}_{}_{}".format(t[0], t[1], t[2])].append(l)
        self.b_inequality = [0] * len(self.A_inequality)
        self.calculated = {}
        self.bounds = []
        self.changed = False
        for i in range(self.no_vars):
            self.bounds.append((0, 1))

    def _calc_offset(self, edge):
        x, y = min(edge), max(edge)
        return int(x * (self.order_val - 1 + self.order_val - x)/2) + y  - x - 1

    def is_uncalculated(self, x, y):
        u, v = min(x, y), max(x, y)
        return not(u in self.calculated and v in self.calculated[u])

    def KnownEdges(self, known_edges):
        for edge in known_edges:
            val = known_edges[edge]
            l = [0] * self.no_vars
            x, y = min(edge), max(edge)
            self.bounds[self._calc_offset(edge)] = [val, val]
            if x not in self.calculated:
                self.calculated[x] = set([y])
            else:
                self.calculated[x].add(y)
            l[self._calc_offset(edge)] = 1
            self.b_equality.append(known_edges[edge])
            self.A_equality.append(l)

    def eqn_maker(self, triples):
        c = [0] * self.no_vars
        for t in triples:
            x, y = min(t[:2]), max(t[:2])
            c[self._calc_offset([t[0], t[1]])] = t[2]
        return c

    def update(self, edge, val):
        l = [0] * self.no_vars
        x, y = min(edge), max(edge)
        if x not in self.calculated:
            self.calculated[x] = set([y])
        else:
            self.calculated[x].add(y)
        l[self._calc_offset(edge)] = 1
        self.b_equality.append(val)
        self.A_equality.append(l)
        self.bounds[self._calc_offset(edge)] = [val, val]
        for node in range(self.order_val):
            if node == edge[0] or node == edge[1]:
                continue
            if (not self.is_uncalculated(node, edge[0])) and (not self.is_uncalculated(node, edge[1])):
                self.delete_inequality(node, edge[0], edge[1])

    def solve_inequality(self, c):
        #res = linprog(c, A_ub=self.A_inequality, b_ub=self.b_inequality, bounds=self.bounds)
        #return res
        if self.changed:
            self.changed = False
            self.A_inequality = []
            for val in self.A_inequality_dict.values():
                self.A_inequality.append(val[0])
                self.A_inequality.append(val[1])
                self.A_inequality.append(val[2])
            self.b_inequality = [0] * len(self.A_inequality)
        if len(self.A_equality) > 0:
            return linprog(c, A_ub=self.A_inequality, b_ub=self.b_inequality, A_eq=self.A_equality, b_eq=self.b_equality, bounds=[(0.0, 1.0)])
        else:
            return linprog(c, A_ub=self.A_inequality, b_ub=self.b_inequality, bounds=[(0.0, 1.0)])

    def delete_inequality(self, a, b, c):
        self.changed = True
        d = [a, b, c]
        d.sort()
        a, b, c = d[0], d[1], d[2]
        del self.A_inequality_dict["{}_{}_{}".format(a, b, c)]

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
        for i in range(self.order-1):
            if i % 1000 == 0:
                print("Number of Edges found by primes: {}".format(i))
            candidate_set = []
            """
            for r in removed:
                for n in set_nodes:
                    i = 0
                    add_edge = True
                    while i < len(candidate_set):
                        soln_min = self.plug_in_oracle.solve_inequality(self.plug_in_oracle.eqn_maker([(r, n, 1), (candidate_set[i][0], candidate_set[i][1], -1)]))
                        soln_max = self.plug_in_oracle.solve_inequality(self.plug_in_oracle.eqn_maker([(r, n, -1), (candidate_set[i][0], candidate_set[i][1], 1)]))
                        #print("{}-{}:min={},max={}".format([r, n], candidate_set[i], soln_min.fun, -soln_max.fun))
                        if -soln_min.fun < 0:
                            add_edge = False
                            #break
                        if -soln_max.fun < 0:
                            print(candidate_set[i])
                            del candidate_set[i]
                        else:
                            i += 1
                        # soln_max.x[self.plug_in_oracle._calc_offset()]
                    if add_edge:
                        candidate_set.append((r, n))
            """
            for r in removed:
                for n in set_nodes:
                    candidate_set.append((r, n))
            for r in removed:
                for n in set_nodes:
                    i = 0
                    add_edge = True
                    while i < len(candidate_set):
                        if min(r,n) == min(candidate_set[i]) and max(r, n) == max(candidate_set[i]):
                            i += 1
                            continue
                        soln_max = self.plug_in_oracle.solve_inequality(self.plug_in_oracle.eqn_maker([(r, n, -1), (candidate_set[i][0], candidate_set[i][1], 1)]))
                        if not soln_max.success:
                            pdb.set_trace()
                        if -soln_max.fun <= 0:
                            del candidate_set[i]
                        else:
                            i += 1
            min_dist = 1
            min_edge = None
            print(candidate_set)
            for c in candidate_set:
                if self.plug_in_oracle.is_uncalculated(c[0], c[1]):
                    dist = self.oracle(c[0], c[1])
                    print("{}-{}:{}".format(c[0], c[1], dist))
                    self.plug_in_oracle.update((c[0], c[1]), dist)
                else:
                    soln_opt = self.plug_in_oracle.eqn_maker([(c[0], c[1], 1)])
                    soln_lp = self.plug_in_oracle.solve_inequality(soln_opt)
                    dist = soln_lp.fun
                if dist < min_dist:
                    min_dist = dist
                    min_edge = c
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

def prims_DFT(order_val, oracle, timer):

    print("Experiment Starting Sasha Wang (Prims)\n")

    global full_mat, count
    # g = {}

    timer.start()
    oracle_plugin = DirFeasibilityTest(order_val)

    # oracle = flicker_oracle()
    p = PrimsDFT(order_val, oracle, oracle_plugin)
    p.mst(0)
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
        "COUNT DFT: {}, Time(total): {}, Time(SP): \n\n".format(count, timer.time_elapsed))


def helper_prims_plugin(order_val, scheme_chooser_list=1, scale_k=1, oracle_chooser=2, bounder=False, three=False):
    global count

    oracle = None
    if oracle_chooser == 1:
        oracle = flicker_oracle()
    elif oracle_chooser == 2:
        oracle = SFOracle(debug=False)
    elif oracle_chooser == 3:
        oracle = UCIUrbanOracle(debug=False)
    elif oracle_chooser == 4:
        oracle = TestOracle()
        # oracle = SFOracle(debug=True)
    assert oracle is not None
    # oracle_plugin = NodeLandMarkRandom({}, k, oracle, order_val)
    # print("Priming Cost: {}\nTime for priming: {}\n".format(count, oracle_plugin.time2prime))

    # pr = vanila_prims(order_val, time_waste_oracle)
    pr = vanila_prims(order_val, oracle)
    timer = Timer()
    timer.start()
    pr.mst(0)
    timer.end()
    base_algo_results = "COUNT Without Plugin(Vanila) " + str(count) + "\nTime(Vanila) " + str(
        timer.time_elapsed) + "\n"
    print(base_algo_results, "\n\n")

    prims_SW({}, pr, order_val, timer, oracle, bounder=bounder)
    date_time_printer()

    new = prims_DFT(order_val, oracle, timer)
helper_prims_plugin(order_val=16, scheme_chooser_list=1, scale_k=1, oracle_chooser=2, bounder=False, three=False)
