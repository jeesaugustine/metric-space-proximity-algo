import operator

class Kruskals:
    def __init__(self, order, oracle, plug_in_oracle):
        # lb, ub = plug_in_oracle(u, v)[0]
        # actual_distance = oracle(u, v)
        self.order = order
        self.parent = list(range(self.order))
        self.counts = [1] * self.order
        self.oracle = oracle
        self.plug_in_oracle = plug_in_oracle
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
        if self.counts[index_1] > self.counts[index_2]:
            self.parent[index_2] = index_1
            self.counts[index_2] += self.counts[index_1]
        else:
            self.parent[index_1] = index_2
            self.counts[index_1] += self.counts[index_2]

    def mst(self):
        distance_tuple = []
        for edge_i in range(self.order):
            for edge_j in range(edge_i + 1, self.order):
                distance_tuple.append([self.plug_in_oracle.lookup(edge_i, edge_j)[0], (edge_i, edge_j)])
        distance_tuple = sorted(distance_tuple, key=operator.itemgetter(0))
        total = 0
        total_path = 0
        while total < self.order - 1:
            cur_edge = distance_tuple[0]
            if len(distance_tuple) == 1:
                self.mst_dict[cur_edge[1]] = cur_edge[0]
                total += 1
                continue
            upper_bound = self.plug_in_oracle.lookup(*cur_edge[1])[1]
            if upper_bound <= distance_tuple[1][0]:
                distance_tuple.pop(0)
                x, y = cur_edge[1]
                r1 = self._find(x)
                r2 = self._find(y)
                if r1 != r2:
                    self._union(r1, r2)
                    index = 0
                    while index < len(distance_tuple):
                        x, y = distance_tuple[index][1]
                        r1 = self._find(x)
                        r2 = self._find(y)
                        if r1 == r2:
                            distance_tuple.pop(index)
                        else:
                            index += 1
                    self.mst_dict[cur_edge[1]] = cur_edge[0]
                    total_path += cur_edge[0]
                    total += 1
            else:
                # Should always be true
                to_update = None
                if self.plug_in_oracle.is_uncalculated(*cur_edge[1]):
                    to_update = cur_edge
                elif self.plug_in_oracle.is_uncalculated(*distance_tuple[1][1]):
                    to_update = distance_tuple[1]
                else:
                    distance_tuple.pop(0)
                    continue
                actual = self.oracle(*to_update[1])
                self.plug_in_oracle.update(to_update[1], actual)
                for index, entry in enumerate(distance_tuple):
                   distance_tuple[index][0] = self.plug_in_oracle.lookup(*entry[1])[0]
                distance_tuple = sorted(distance_tuple, key=operator.itemgetter(0))
        print(total_path)
        self.mst_path_length = total_path
