import heapq

class Kruskals:
    def __init__(self, order, oracle):
        self.order = order
        self.parent = list(range(self.order))
        self.counts = [1] * self.order
        self.oracle = oracle
        self.mst_dict = dict()

    def _find(self, index):
        check_list = []
        cur_index = index
        while self.parent[cur_index] != cur_index:
            check_list.append(cur_index)
            cur_index = self.parent[cur_index]
        for index in check_list:
            self.parent[index] = cur_index
        return cur_index

    def _union(self, index_1, index_2):
        # print("Edges are: ({}, {})".format(index_1, index_2))
        if self.counts[index_1] > self.counts[index_2]:
            self.parent[index_2] = index_1
            self.counts[index_2] += self.counts[index_1]
        else:
            self.parent[index_1] = index_2
            self.counts[index_1] += self.counts[index_2]

    def mst(self):
        distance_tuple = []
        for edge_i in range(self.order):
            for edge_j in range(edge_i, self.order):
                distance_tuple.append((self.oracle(edge_i, edge_j), (edge_i, edge_j)))
        heapq.heapify(distance_tuple)
        total = 0
        total_path = 0
        while total < self.order - 1:
            cur_edge = heapq.heappop(distance_tuple)
            x, y = cur_edge[1]
            r1 = self._find(x)
            r2 = self._find(y)
            if r1 != r2:
                self._union(r1, r2)
                self.mst_dict[cur_edge[1]] = cur_edge[0]
                total_path += cur_edge[0]
                total += 1
        print(total_path)
        self.mst_path_length = total_path