from pulp import *
from random import choices
from graph_maker import NlogNGraphMaker
from helper import _get_matrix_from_adj_dict
from bottom_up_tree_sample_min_max import BottomUPTree
from lower_bound_tree import LBTree
import numpy as np


class LP_Solver:
    def _conv(self, x, y):
        _min = min(x, y)
        _max = max(x, y)
        if x == y:
            return -1
        else:
            return self.n * _min - int(_min * (_min + 1) / 2) + _max - _min - 1

    def __init__(self, n, sampling=True, samples=5):
        self.n = n
        self.edges = {}
        self.variables = pulp.LpVariable.dicts('x', range(int(n*(n-1)/2)), lowBound=0.0, upBound=1.0)
        self.optimization = pulp.LpProblem("test problem",pulp.LpMaximize)
        self.optimization += pulp.lpSum([self.variables[0]]), "objective"
        if sampling:
            l = list(range(n))
            for i in range(n):
                for j in range(i+1,n):
                    sample = choices(l, k=samples)
                    for k in sample:
                        if k == i or k == j:
                            continue
                        self.optimization += self.variables[self._conv(i, j)] + self.variables[self._conv(i, k)] - \
                                             self.variables[self._conv(j, k)] >= 0
                        self.optimization += self.variables[self._conv(i, j)] + self.variables[self._conv(j, k)] - \
                                             self.variables[self._conv(i, k)] >= 0
                        self.optimization += self.variables[self._conv(i, k)] + self.variables[self._conv(j, k)] - \
                                             self.variables[self._conv(i, j)] >= 0
            return
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1, n):
                    self.optimization += self.variables[self._conv(i, j)] + self.variables[self._conv(i, k)] - \
                                         self.variables[self._conv(j, k)] >= 0
                    self.optimization += self.variables[self._conv(i, j)] + self.variables[self._conv(j, k)] - \
                                         self.variables[self._conv(i, k)] >= 0
                    self.optimization += self.variables[self._conv(i, k)] + self.variables[self._conv(j, k)] - \
                                         self.variables[self._conv(i, j)] >= 0

    def solv(self, obj): #[{"edge": val, "coefficient": val}]
        self.optimization.setObjective(pulp.lpSum([item["coefficient"] * self.variables[item["edge"]] for item in obj]))
        self.optimization.solve()
        return self.optimization.objective.value()

    def add_constraint(self, x, y, value):
        _min = min(x, y)
        _max = max(x , y)
        if _min not in self.edges:
            self.edges[_min] = {}
        self.edges[_min][_max] = value
        self.optimization += self.variables[self._conv(_min, _max)] == value

    def lookup(self, x, y):
        _min = min(x, y)
        _max = max(x , y)
        if _min in self.edges and _max in self.edges[_min]:
            return self.edges[_min][_max]
        return [-1 * self.solv([{"edge": self._conv(_min, _max), "coefficient": -1}]), 
                self.solv([{"edge": self._conv(_min, _max), "coefficient": 1}])]


if __name__ == '__main__':
    order_val = 32

    graph_maker = NlogNGraphMaker(order_val)
    g = graph_maker.get_nlogn_edges()
    full_mat = graph_maker.out
    g_mat = _get_matrix_from_adj_dict(g, order_val)
    edges = graph_maker.get_nlogn_edges()
    lp = LP_Solver(order_val)
    for e in edges.keys():
        lp.add_constraint(e[0], e[1], edges[e])
    lp1 = LP_Solver(order_val, sampling=False)
    for e in edges.keys():
        lp1.add_constraint(e[0], e[1], edges[e])
    out_ub = 0
    out_lb = 0
    out_lb1 = 0
    lb_mat = np.copy(g_mat)
    bu_tree = BottomUPTree(g, order_val)
    bu_tree.build_tree()
    tree = bu_tree.tree
    lb = LBTree(tree, lb_mat)
    import time as t
    timer_act = 0
    timer_drop = 0
    for i in range(order_val):
        print(i)
        for j in range(i+1,order_val):
            # print(j)
            if not (i, j) in edges:
                start_time = t.time()
                ans_app = lp.lookup(i, j)
                end_time = t.time()
                ans = lp1.lookup(i, j)
                end_time1 = t.time()
                timer_drop += end_time - start_time
                timer_act += end_time1 - end_time
                out_lb += ans[0] - ans_app[0]
                out_ub += ans_app[1] - ans[1]
                out_lb1 += ans[0] - lb.lb_matrix[i][j]
    print("LB", out_lb)
    print("UB", out_ub)

    print("Time Act: ", timer_act / (order_val * (order_val - 1) / 2 - len(edges)))
    print("Time Loose: ", timer_drop / (order_val * (order_val - 1) / 2 - len(edges)))
    print("LB Avg: ", out_lb/(order_val*(order_val-1)/2-len(edges)))
    print("UB Avg: ", out_ub/(order_val*(order_val-1)/2-len(edges)))
    print("LB Avg LBT: ", out_lb1 / (order_val * (order_val - 1) / 2 - len(edges)))

