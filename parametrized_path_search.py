from sparse_matrix import SparseMatrix
from dijkstra import Dijkstra
from helper import _get_dbl_level_dict
import numpy as np
import time

class ParamTriSearch:
    def __init__(self, max_path_length, sw_ub):
        self.max_path_length = max_path_length
        self.sw_ub = sw_ub
        self.sparse_matrix = None
        self.lb_matrix = []
        self.ub_matrix = []
        self.search_started = []
        self.uncalculated = {}
        assert self.max_path_length >= 2
        self.update_time = 0

    def lookup(self, x, y, fake=False):
        if fake:
            return [self.lb_matrix[x][y], self.ub_matrix[x][y]]
        if not(self.search_started[x] or self.search_started[y]):
            #self._bfs(x)
            self._update(x, y)
            self.search_started[x] = True
        return [self.lb_matrix[x][y], self.ub_matrix[x][y]]

    def store(self, distance_hash, n):
        self.order = n
        self._dbl_lvl_dict = _get_dbl_level_dict(distance_hash)
        sp_dijkstra = Dijkstra(self._dbl_lvl_dict, self.order)
        for i in range(self.order):
            self.ub_matrix.append([1]*self.order)
        for i in range(self.order):
            nodes = list(range(self.order))
            sp = sp_dijkstra.shortest_path(nodes, i)
            for index in range(self.order):
                # distance, node, parent
                self.ub_matrix[i][index] = sp[index][0]
                self.ub_matrix[index][i] = sp[index][0]
        self.search_started = [False] * n
        for i in range(n):
            self.lb_matrix.append([0] * n)
            if n - i - 1 != 0:
                self.uncalculated[i] = set(range(i + 1, n))
        for k in distance_hash.keys():
            x, y = k
            self.lb_matrix[x][y] = distance_hash[k]
            self.lb_matrix[y][x] = distance_hash[k]
            self.uncalculated[min(x, y)].remove(max(x, y))
            if len(self.uncalculated[min(x, y)]) == 0:
                del self.uncalculated[min(x, y)]
        self.sparse_matrix = SparseMatrix(distance_hash, n)
        if self.max_path_length == 2:
            for i in range(n):
                for j in range(i+1, n):
                    self._update(i, j)
        else:
            for i in range(n):
                self._dfs(i)

    def update(self, edge, val):
        start = time.time()
        x, y = edge
        # try:
        #     assert not np.any(np.array(self.sw_ub) - np.array(self.ub_matrix) > 0.000001)
        # except AssertionError:
        #     print('Something hit me ')
        self.lb_matrix[x][y] = self.lb_matrix[y][x] = val
        self.ub_matrix[x][y] = self.ub_matrix[y][x] = val
        self.search_started = [False] * len(self.lb_matrix)
        self.uncalculated[min(x, y)].remove(max(x, y))
        if len(self.uncalculated[min(x, y)]) == 0:
            del self.uncalculated[min(x, y)]
        # try:
        #     assert not np.any(np.array(self.sw_ub) - np.array(self.ub_matrix) > 0.000001)
        # except:
        #     print('a')
        self._sw_lb_update(edge, val)
        # try:
        #     assert not np.any(np.array(self.sw_ub) - np.array(self.ub_matrix) > 0.000001)
        # except AssertionError:
        #     print('Something hit me ')
        end = time.time()
        self.update_time += end - start

    def _bfs(self, node):
        queue = []
        # (node_index, max, path_length)
        last = (node, 0, 0)
        queue.append(last)
        cur_path_length = 0
        while len(queue) > 0:
            head = queue.pop(0)
            distance, neighbours = self.sparse_matrix.get_row_data(head[0])
            for i in range(len(neighbours)):
                cur_node = neighbours[i]
                max_val = max(head[1], distance[i])
                path_val = head[2] + distance[i]
                self.lb_matrix[node][cur_node] = self.lb_matrix[cur_node][node] = max(
                    self.lb_matrix[cur_node][node],
                    2 * max_val - path_val
                )
                queue.append((cur_node, max_val, path_val))
            if head == last:
                if cur_path_length == self.max_path_length or len(queue) == 0:
                    return
                else:
                    last = queue[-1]
                    cur_path_length += 1

    def _update(self, x, y):
        if self.max_path_length > 2:
            self._dfs(x)
        else:
            self._calculate(x, y)

    def _calculate(self, x, y):
        _, n1 = self.sparse_matrix.get_row_data(x)
        _, n2 = self.sparse_matrix.get_row_data(y)
        common = set(n1).intersection(set(n2))
        for c in common:
            d1 = self.sparse_matrix.get_element(x, c)
            d2 = self.sparse_matrix.get_element(y, c)
            if d1 > d2:
                _max = d1
                _min = d2
            else:
                _max = d2
                _min = d1
            self.lb_matrix[x][y] = self.lb_matrix[x][y] = max(self.lb_matrix[x][y], _max - _min)

    def _dfs(self, node):
        self._dfs_recursive(node, node, 0, 0, {node}, 0)

    def _dfs_recursive(self, start_node, node, max_edge, path_length, visited, depth):
        distance, neighbours = self.sparse_matrix.get_row_data(node)
        for i in range(len(neighbours)):
            cur_node = neighbours[i]
            if cur_node in visited:
                continue
            max_val = max(max_edge, distance[i])
            path_val = path_length + distance[i]
            self.lb_matrix[start_node][cur_node] = self.lb_matrix[cur_node][start_node] = max(
                self.lb_matrix[start_node][cur_node],
                2 * max_val - path_val
            )
            new_visited = visited.union({cur_node})
            if depth < self.max_path_length:
                self._dfs_recursive(start_node, cur_node, max_val, path_val, new_visited, depth+1)

    def is_uncalculated(self, x, y):
        return min(x, y) in self.uncalculated and max(x, y) in self.uncalculated[min(x, y)]

    def _sw_lb_update(self, edge, d):
        x, y = edge
        for i, inds in self.uncalculated.items():
            for j in inds:
                self.ub_matrix[i][j] = self.ub_matrix[j][i] = min(self.ub_matrix[j][i],
                                                                  self.ub_matrix[i][x] + d + self.ub_matrix[y][j],
                                                                  self.ub_matrix[i][y] + d + self.ub_matrix[x][j])
