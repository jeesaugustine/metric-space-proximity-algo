from dijkstra import Dijkstra
from helper import _get_dbl_level_dict
from sortedcontainers import SortedSet
import time


class IntersectTriSearch:
    def __init__(self, g, order_val):
        self.G = g
        # self.noNodes = len(set([node for nodes in list(self.G.keys()) for node in nodes]))
        self.noNodes = order_val
        self.vertices = [i for i in range(1, self.noNodes + 1)]
        self.adj_list = dict(zip(range(self.noNodes), [SortedSet([]) for i in range(self.noNodes)]))
        self.get_adj_list(graph=self.G)
        self.sp_time = 0
        self.lookup_count = 0

    def get_adj_list(self, graph):
        for (u, v) in graph:
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def lookup(self, x, y):
        if x == y:
            return [0, 0]
        self.lookup_count += 1
        u, v = min((x, y)), max((x, y))
        if self.G.get((u, v), -1) > -1:
            return [self.G[(u, v)], self.G[(u, v)]]
        isect = self.get_intersection(u, v)
        # start_timer = time.time()
        # self._dbl_lvl_dict = _get_dbl_level_dict(self.G)
        # self.sp_dijkstra = Dijkstra(self._dbl_lvl_dict, self.noNodes)
        # sp = self.sp_dijkstra.shortest_path(list(range(self.noNodes)), x)
        # end_timer = time.time()
        # self.sp_time += end_timer - start_timer
        minim = 1
        maxim = 0
        for each in isect:
            cur_minim = self.G[(min(u, each), max(u, each))] + self.G[(min(v, each), max(v, each))]
            if cur_minim <= minim:
                minim = cur_minim
            cur_max = abs(self.G[(min(u, each), max(u, each))] - self.G[(min(v, each), max(v, each))])
            if cur_max >= maxim:
                maxim = cur_max
        # return [maxim, sp[y][0]]
        return [maxim, minim]

    def get_intersection(self, u, v):
        return list(self.adj_list[u].intersection(self.adj_list[v]))
        # return list(set(self.adj_list[u]) & set(self.adj_list[v]))

    def update(self, edge, val):
        u, v = min(edge), max(edge)
        self.G[(u, v)] = val
        self.adj_list[u].add(v)
        self.adj_list[v].add(u)
        # self.adj_list[u].append(v)
        # self.adj_list[v].append(u)

    def is_uncalculated(self, x, y):
        return (x != y) and (not (((x, y) in self.G) or ((y, x) in self.G)))


def read_graph(path_to_graph, filename):
    import pickle
    import os
    g = pickle.load(open(os.path.join(path_to_graph, filename), 'rb'))
    return g


if __name__ == '__main__':
    # If you have a graph that is stored you can input that here and call get_adj_list(graph); Else you can skip it
    g = read_graph('', 'toygraph_4_4.pkl')

    its = IntersectTriSearch(g=g, )

    print(its.lookup((2, 4)))
    its.update((1, 3), 0.3)
    print(its.G, '\n', its.adj_list)
