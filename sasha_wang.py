import time

class SashaWang():
    def __init__(self):
        self.matrix = []
        self.lb_matrix = []
        self.ub_matrix = []
        self.uncalculated = {}
        self.update_time = 0

    def _sw_store(self, dist_hash, n):
        for i in range(n):
            self.matrix.append([-1] * n)
            self.lb_matrix.append([0] * n)
            self.ub_matrix.append([1] * n)
            self.ub_matrix[i][i] = 0
            if n - i - 1 != 0:
                self.uncalculated[i] = set(range(i + 1, n))
        for k in dist_hash.keys():
            x, y = k
            if x == y:
                continue
            self.matrix[x][y] = dist_hash[k]
            self.matrix[y][x] = dist_hash[k]
            self.lb_matrix[x][y] = self.matrix[x][y]
            self.lb_matrix[y][x] = self.matrix[y][x]
            self.ub_matrix[x][y] = self.matrix[x][y]
            self.ub_matrix[y][x] = self.matrix[y][x]
            self.uncalculated[min(x, y)].remove(max(x, y))
            if len(self.uncalculated[min(x, y)]) == 0:
                del self.uncalculated[min(x, y)]
        # start = time.time()
        if len(dist_hash) > 2:
            self._sw_one_shot()
        # end = time.time()
        # print(end-start)

    def _sw_one_shot(self):
        n = len(self.matrix)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    self.lb_matrix[i][j] = self.lb_matrix[j][i] = max(self.lb_matrix[i][j], self.lb_matrix[i][k] - self.ub_matrix[k][j],
                                          self.lb_matrix[j][k] - self.ub_matrix[k][i])
                    self.ub_matrix[i][j] = self.ub_matrix[j][i] = min(self.ub_matrix[i][j], self.ub_matrix[i][k] + self.ub_matrix[k][j])

    def _sw_lb_update(self, edge, i, j):
        x, y = edge
        d = self.matrix[x][y]
        self.lb_matrix[i][j] = self.lb_matrix[j][i] = max(
            self.lb_matrix[i][j],
            d - self.ub_matrix[i][x] - self.ub_matrix[j][y],
            d - self.ub_matrix[i][y] - self.ub_matrix[j][x],
            self.lb_matrix[i][x] - d - self.ub_matrix[j][y],
            self.lb_matrix[i][y] - d - self.ub_matrix[j][x],
            self.lb_matrix[j][x] - d - self.ub_matrix[i][y],
            self.lb_matrix[j][y] - d - self.ub_matrix[i][x]
        )
        self.ub_matrix[i][j] = self.ub_matrix[j][i] = min(self.ub_matrix[j][i],
                self.ub_matrix[i][x] + d + self.ub_matrix[y][j],
                self.ub_matrix[i][y] + d + self.ub_matrix[x][j])

    def _sw_update(self, edge, val):
        start = time.time()
        x, y = edge
        self.matrix[x][y] = self.matrix[y][x] = self.lb_matrix[x][y] = self.lb_matrix[y][x] = self.ub_matrix[x][y] = self.ub_matrix[y][x] = val
        self.uncalculated[min(x, y)].remove(max(x, y))
        if len(self.uncalculated[min(x, y)]) == 0:
            del self.uncalculated[min(x, y)]
        for i, inds in self.uncalculated.items():
            for j in inds:
                self._sw_lb_update(edge, i, j)
        end = time.time()
        self.update_time += end - start

    def lookup(self, x, y):
        if x == y:
            return [0, 0]
        return [self.lb_matrix[x][y], self.ub_matrix[x][y]]

    def store(self, distance_hash, n):
        self._sw_store(distance_hash, n)

    def update(self, edge, val):
        self._sw_update(edge, val)

    def is_uncalculated(self, x, y):
        return (x != y) and (min(x, y) in self.uncalculated and max(x, y) in self.uncalculated[min(x, y)])
