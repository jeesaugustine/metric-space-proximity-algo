from random import choices
import copy
from helper import _get_dbl_level_dict
from dijkstra import Dijkstra
import time


class LandMark:
    def __init__(self, edge_list, order_val, k):
        self.edge_list = edge_list
        self.k = k
        self.order_val = order_val
        self.dbl_dict = _get_dbl_level_dict(self.edge_list)
        self.d = Dijkstra(self.dbl_dict, self.order_val)
        self.landmarks = None
        self.dijk_dict = {}

    def get_edge_landmark(self):
        choice = []
        edge_list = copy.copy(self.edge_list)
        edge, value = list(edge_list.keys()), list(edge_list.values())
        for each in range(self.k):
            c = choices(edge, value)[0]
            del edge_list[tuple(c)]
            choice.append((min(c[0], c[1]), max(c[0], c[1])))
            edge, value = list(edge_list.keys()), list(edge_list.values())
        self.landmarks = copy.copy(choice)

    def landmark_dijk(self):
        order_val = list(range(self.order_val))
        self.dijk_dict = {}
        for edge in self.landmarks:
            x, y = edge
            for i in [x, y]:
                if i not in self.dijk_dict:
                    dij = self.d.shortest_path(order_val, i)
                    self.dijk_dict[i] = []
                    for j in range(self.order_val):
                        self.dijk_dict[i].append(dij[j][0])

    def look_up(self, x, y):
        _min = min(x, y)
        _max = max(x, y)
        if _min in self.dbl_dict and _max in self.dbl_dict[_min]:
            return [self.dbl_dict[_min][_max], self.dbl_dict[_min][_max]]
        lb = 0
        ub = 1
        for e in self.landmarks:
            i, j = e
            ix, iy, jx, jy = self.dijk_dict[i][x], self.dijk_dict[i][y], self.dijk_dict[j][x], self.dijk_dict[j][y]
            landmark_len = self.dbl_dict[min(i, j)][max(i, j)]
            ub = min(ub, landmark_len + ix + iy, landmark_len + jx + jy)
            lb = max(lb, landmark_len - ix - jy, landmark_len - iy - jx)
        return [lb, ub]

    def mat_look_up(self):
        lb_mat = []
        ub_mat = []
        for i in range(self.order_val):
            lb_mat.append([0] * self.order_val)
            ub_mat.append([1] * self.order_val)
        try_time = 0
        for i in range(self.order_val):
            for j in range(i + 1, self.order_val):
                if i in self.dbl_dict and j in self.dbl_dict[i]:
                    lb_mat[i][j], ub_mat[i][j] = self.dbl_dict[i][j], self.dbl_dict[i][j]
                for e in self.landmarks:
                    x, y = e
                    start_inner = time.time()
                    ix, iy, jx, jy = self.dijk_dict[x][i], self.dijk_dict[y][i], self.dijk_dict[x][j], \
                                     self.dijk_dict[y][j]
                    end_inner = time.time()
                    try_time += end_inner - start_inner
                    landmark_len = self.dbl_dict[x][y]
                    if ix + jx < iy + jy:
                        if landmark_len + ix + jx < ub_mat[i][j]:
                            ub_mat[i][j] = landmark_len + ix + jx
                    elif landmark_len + iy + jy < ub_mat[i][j]:
                        ub_mat[i][j] = landmark_len + iy + jy
                    if ix + jy < iy + jx:
                        if lb_mat[i][j] < landmark_len - ix - jy:
                            lb_mat[i][j] = landmark_len - ix - jy
                    elif lb_mat[i][j] < landmark_len - iy - jx:
                        lb_mat[i][j] = landmark_len - iy - jx
        end = time.time()
        # print(end-start, try_time)
        return [lb_mat, ub_mat]

    def update(self, edge, val):
        self.edge_list[edge] = val
        if edge[0] < edge[1]:
            _min = edge[0]
            _max = edge[1]
        else:
            _min = edge[1]
            _max = edge[0]
        self.dbl_dict[_min][_max] = val
        sp = [self.d.shortest_path(list(range(self.order_val)), _min),
              self.d.shortest_path(list(range(self.order_val)), _max)]
        visited = set()
        for e in self.landmarks:
            e1, e2 = e
            for landmark in [e1, e2]:
                if landmark in visited:
                    continue
                visited.add(landmark)
                for i in range(self.order_val):
                    cur = self.dijk_dict[landmark][i]
                    min_L = sp[0][landmark]
                    max_L = sp[1][landmark]
                    min_i = sp[0][i]
                    max_i = sp[1][i]
                    if min_L + max_i < min_i + max_L:
                        if min_L + max_i + val < cur:
                            self.dijk_dict[landmark][i] = min_L + max_i + val
                    elif min_i + max_L + val < cur:
                        self.dijk_dict[landmark][i] = min_i + max_L + val
