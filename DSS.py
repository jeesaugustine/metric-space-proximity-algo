import random
from dijkstra import Dijkstra
from helper import _get_dbl_level_dict


def select_the_new_landmark(sum_node_to_landmark):
    max_key = max(sum_node_to_landmark, key=sum_node_to_landmark.get)

    return max_key


class DSS:
    def __init__(self, g, order_val):
        self.G = g
        self.noNodes = order_val
        self.vertices = [i for i in range(self.noNodes)]
        self._dbl_lvl_dict = _get_dbl_level_dict(self.G, order_val)

    def lookup(self, x, y):
        u, v = min(x, y), max(x, y)
        if self.G.get((u, v), -1) > -1:
            return [self.G[(u, v)], self.G[(u, v)]]
        ub = 1
        lb = 0
        dijk_obj = Dijkstra(self._dbl_lvl_dict, self.noNodes)
        x_sp = dijk_obj.shortest_path(self.vertices, x)
        y_sp = dijk_obj.shortest_path(self.vertices, y)
        if y not in x_sp:
            return [lb, ub]

        if x_sp[y][0] < ub:
            ub = x_sp[y][0]

        for (a, b) in self.G.items():
            s, t = a
            if s in x_sp and t in x_sp:
                lb = max(lb, b-x_sp[s][0]-y_sp[t][0], b-x_sp[t][0]-y_sp[s][0])

        return [lb, ub]

    def update(self, edge, val):
        u, v = min(edge), max(edge)
        self.G[(u, v)] = val
        self._dbl_lvl_dict[u][v] = val
        self._dbl_lvl_dict[v][u] = val

    def is_uncalculated(self, x, y):
        return not (((x, y) in self.G) or ((y, x) in self.G))
