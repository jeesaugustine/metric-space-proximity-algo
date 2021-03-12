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
        # print("Edges are: ({}, {})".format(index_1, index_2))
        if self.counts[index_1] > self.counts[index_2]:
            self.parent[index_2] = index_1
            self.counts[index_2] += self.counts[index_1]
        else:
            self.parent[index_1] = index_2
            self.counts[index_1] += self.counts[index_2]

    def mst(self):
        count = 0
        total = 0
        # import pdb
        # pdb.set_trace()
        while count < self.order-1:
            # if count == 13:
            #     import pdb
            #     pdb.set_trace()
            candidate_set_mst = self.get_candidate_set_mst()
            if count != 0:
                # print("Candidate set: {}".format(candidate_set_mst))
                pass
            cur_candidate = candidate_set_mst[0][1]
            del candidate_set_mst[0]
            if self.plug_in_oracle.is_uncalculated(*cur_candidate):
                dist_cur_candidate = self.oracle(*cur_candidate)
                self.plug_in_oracle.update(cur_candidate, dist_cur_candidate)
            else:
                dist_cur_candidate = self.plug_in_oracle.lookup(*cur_candidate)[0]
            for candidate in candidate_set_mst:
                bound = self.plug_in_oracle.lookup(*candidate[1])

                if bound[0] > dist_cur_candidate:
                    continue
                if self.plug_in_oracle.is_uncalculated(*candidate[1]):
                    dist = self.oracle(*candidate[1])
                    self.plug_in_oracle.update(candidate[1], dist)
                else:
                    dist = self.plug_in_oracle.lookup(*candidate[1])[0]
                if dist < dist_cur_candidate:
                    dist_cur_candidate = dist
                    cur_candidate = candidate[1]
            self.mst_dict[cur_candidate[0]] = cur_candidate[1]
            # print("Edges are: ({}, {})".format(cur_candidate[0], cur_candidate[1]))
            self._union(self._find(cur_candidate[0]), self._find(cur_candidate[1]))
            count += 1
            total += dist_cur_candidate
        self.mst_path_length = total



    def get_candidate_set_mst(self):
        candidate_set_mst = []
        max_lb = 0
        min_ub = 1
        for i in range(self.order):
            for j in range(i + 1, self.order):
                if self._find(i) == self._find(j):
                    continue
                dist_i_j = self.plug_in_oracle.lookup(i, j)
                if dist_i_j[0] > min_ub:
                    continue
                if dist_i_j[1] < max_lb:
                    candidate_set_mst = list(filter(lambda x: x[0][0] < dist_i_j[1], candidate_set_mst))
                    max_lb = 0
                    min_ub = 1
                    for candidate in candidate_set_mst:
                        max_lb = max(max_lb, candidate[0][0])
                        min_ub = min(min_ub, candidate[0][1])
                max_lb = max(max_lb, dist_i_j[0])
                min_ub = min(min_ub, dist_i_j[1])
                candidate_set_mst.append((dist_i_j, (i, j)))
        return candidate_set_mst
