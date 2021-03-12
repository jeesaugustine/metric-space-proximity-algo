import math
import random


class NlogNGraphMaker:
    def __init__(self, n, d=2):
        random.seed(100)
        self.out = None
        self.n = n
        self.d = d
        self.get_matrix()
        self.neighbours = {}
        self.visited = [False] * self.n
        self._internal = {}
        assert self.out is not None
        self.nlogn = math.ceil(self.n * math.log(self.n, 2))

    def get_matrix(self):
        import numpy as np
        # print("Graph Maker")
        assert self.n is not None
        s = np.random.uniform(0, 0.5, (self.n, self.d))
        for i in range(s.shape[0]):
            x = np.linalg.norm(s - s[i, :], axis=1).reshape((1, s.shape[0]))
            if self.out is None:
                self.out = x
            else:
                self.out = np.vstack((self.out, x))

    def get_nlogn_edges(self, total=-1):
        n = self.out.shape[0]
        if total < n - 1:
            total = self.nlogn

        val_dict = {}
        x = 0
        while x < total:
            i = int(random.uniform(0, n - 0.00001))
            j = int(random.uniform(0, n - 1.00001))
            if j >= i:
                j += 1
            if (min(i, j), max(i, j)) in val_dict:
                continue
            val_dict[(min(i, j), max(i, j))] = self.out[i][j]
            self._put(i, j, self.out[i][j])
            self._put(j, i, self.out[i][j])
            x += 1
        connected_comp = self._connected_components()
        while len(connected_comp) > 1:
            i = int(random.uniform(0, len(connected_comp) - 0.00001))
            j = int(random.uniform(0, len(connected_comp) - 1.00001))
            if j >= i:
                j += 1
            node_i = random.choice(connected_comp[i])
            node_j = random.choice(connected_comp[j])
            val_dict[(min(node_i, node_j), max(node_i, node_j))] = self.out[node_i][node_i]
            self._put(node_i, node_j, self.out[node_i][node_j])
            self._put(node_j, node_i, self.out[node_i][node_j])
            temp = connected_comp[max(i, j)]
            del connected_comp[max(i, j)]
            connected_comp[min(i, j)] += temp
        return val_dict

    def _put(self, key, value, distance):
        if key in self.neighbours:
            self.neighbours[key].append(value)
        else:
            self.neighbours[key] = [value]

    def _get_neighbours(self, row):
        if row in self.neighbours:
            return self.neighbours[row]
        return []

    def _connected_components(self):
        start_node = self._get_start_node()
        conn_comp = []
        while start_node != -1:
            self._dfs(start_node)
            conn_comp.append(self._internal['connected'])
            start_node = self._get_start_node()
        return conn_comp

    def _get_start_node(self):
        for index, visit in enumerate(self.visited):
            if not visit:
                return index
        return -1

    def _dfs(self, node):
        self.visited[node] = True
        self.visited[node] = True
        self._internal['connected'] = [node]
        self._dfs_recursive(node)

    def _dfs_recursive(self, node):
        neighbours = self._get_neighbours(node)
        for i in range(len(neighbours)):
            cur_node = neighbours[i]
            if self.visited[cur_node]:
                continue
            self.visited[cur_node] = True
            self._internal['connected'].append(cur_node)
            self._dfs_recursive(cur_node)
