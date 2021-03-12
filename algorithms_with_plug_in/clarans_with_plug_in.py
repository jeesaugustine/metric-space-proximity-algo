from math import inf
from random import choice, seed, sample


class CLARANS:
    def __init__(self, n, k, plug_in_oracle, oracle, num_local=None, max_neighbour=None):
        # lb, ub = plug_in_oracle(u, v)[0]
        # actual_distance = oracle(u, v)
        self.plug_in_oracle = plug_in_oracle
        self.n = n
        self.k = k
        self.oracle = oracle
        self.centroids = None
        self.local_centroids = None
        self._clusters = {}
        self._reverse_centroids = {}
        self.min_cost = inf
        self.local_cost = inf
        seed(59)
        if num_local is not None:
            self.num_local = num_local
        else:
            self.num_local = max(self.k, 5)
        if max_neighbour is not None:
            self.max_neighbour = max_neighbour
        else:
            self.max_neighbour = int(0.2 * self.k * (self.n - self.k))
        self.iterate()

    def iterate(self):
        list_k = list(range(self.k))
        set_n = set(range(self.n))
        for i in range(self.num_local):
            self._assign()
            j = 1
            while j <= self.max_neighbour:
                loc = choice(list_k)
                # print(self.local_centroids)
                new_centroid = choice(list(set_n.difference(set(self.local_centroids))))
                tc_jump = self._calculate_diff_potential(loc, new_centroid)
                if tc_jump < 0:
                    # print(self.local_centroids[loc], new_centroid, tc_jump)
                    tc_jump = self._calculate_diff(loc, new_centroid)
                    if tc_jump < 0:
                        self._replace(self.local_centroids[loc], new_centroid)
                        self.local_cost += tc_jump
                        if self.local_cost < self.min_cost:
                            self.min_cost = self.local_cost
                            self.centroids = self.local_centroids
                        j = 0
                j += 1

    def _assign(self):
        self.local_centroids = sample(list(range(self.n)), self.k)
        self._reverse_centroids = {}
        for index, cen in enumerate(self.local_centroids):
            self._reverse_centroids[cen] = index
        self._clusters = {}
        cost = 0
        for i in range(self.n):
            dist = 2
            centroid = -1
            for j in range(self.k):
                if self.plug_in_oracle.is_uncalculated(i, self.local_centroids[j]):
                    dist_orc = self.oracle(i, self.local_centroids[j])
                    self.plug_in_oracle.update((i, self.local_centroids[j]), dist_orc)
                if self.plug_in_oracle.lookup(i, self.local_centroids[j])[1] < dist:
                    dist = self.plug_in_oracle.lookup(i, self.local_centroids[j])[1]
                    centroid = self.local_centroids[j]
            self._clusters[i] = centroid
            cost += dist
        self.local_cost = cost
        if self.local_cost < self.min_cost:
            self.min_cost = self.local_cost
            self.centroids = self.local_centroids

    def _replace(self, old_centroid, new_centroid):
        # print(old_centroid, new_centroid, self.local_centroids, self._reverse_centroids)
        self.local_centroids[self._reverse_centroids[old_centroid]] = new_centroid
        self._reverse_centroids[new_centroid] = self._reverse_centroids[old_centroid]
        del self._reverse_centroids[old_centroid]
        self._clusters[new_centroid] = new_centroid
        # print(self.local_centroids)
        for i in range(self.n):
            cluster_id = self._clusters[i]
            if self.plug_in_oracle.is_uncalculated(i, new_centroid):
                dist_orc = self.oracle(i, new_centroid)
                self.plug_in_oracle.update((i, new_centroid), dist_orc)
            if cluster_id == old_centroid:
                dist = 2
                _id = -1
                for j in range(self.k):
                    if dist > self.plug_in_oracle.lookup(self.local_centroids[j], i)[1]:
                        dist = self.plug_in_oracle.lookup(self.local_centroids[j], i)[1]
                        _id = self.local_centroids[j]
                self._clusters[i] = _id
            elif self.plug_in_oracle.lookup(cluster_id, i)[1] >= self.plug_in_oracle.lookup(i, new_centroid)[1]:
                self._clusters[i] = new_centroid

    def _calculate_diff(self, centroid_index, new_centroid):
        tc_jump = 0
        old_centroid = self.local_centroids[centroid_index]
        for i in range(self.n):
            cluster_id = self._clusters[i]
            if cluster_id == old_centroid:
                dist = 2
                for j in range(self.k):
                    assert not self.plug_in_oracle.is_uncalculated(self.local_centroids[j], i)
                    if j == centroid_index:
                        continue
                    dist = min(dist, self.plug_in_oracle.lookup(self.local_centroids[j], i)[1])
                if self.plug_in_oracle.is_uncalculated(new_centroid, i) and self.plug_in_oracle.lookup(new_centroid, i)[
                    0] < dist:
                    self.plug_in_oracle.update((i, new_centroid), self.oracle(i, new_centroid))
                dist = min(dist, self.plug_in_oracle.lookup(new_centroid, i)[0])
                tc_jump += dist - self.plug_in_oracle.lookup(i, old_centroid)[1]
            elif self.plug_in_oracle.lookup(cluster_id, i)[1] > self.plug_in_oracle.lookup(i, new_centroid)[0]:
                if self.plug_in_oracle.is_uncalculated(new_centroid, i) and \
                        self.plug_in_oracle.lookup(new_centroid, i)[0] < self.plug_in_oracle.lookup(cluster_id, i)[1]:
                    self.plug_in_oracle.update((i, new_centroid), self.oracle(i, new_centroid))
                if self.plug_in_oracle.lookup(new_centroid, i)[0] < self.plug_in_oracle.lookup(cluster_id, i)[1]:
                    tc_jump += self.plug_in_oracle.lookup(i, new_centroid)[1] - \
                               self.plug_in_oracle.lookup(cluster_id, i)[1]
        return tc_jump

    def _calculate_diff_potential(self, centroid_index, new_centroid):
        tc_jump = 0
        old_centroid = self.local_centroids[centroid_index]
        for i in range(self.n):
            if i != old_centroid and i in self.local_centroids:
                continue
            cluster_id = self._clusters[i]
            if cluster_id == old_centroid:
                dist = 2
                for j in range(self.k):
                    if j == centroid_index:
                        continue
                    dist = min(dist, self.plug_in_oracle.lookup(self.local_centroids[j], i)[1])
                dist = min(dist, self.plug_in_oracle.lookup(new_centroid, i)[0])
                tc_jump += dist - self.plug_in_oracle.lookup(i, old_centroid)[1]
            elif self.plug_in_oracle.lookup(cluster_id, i)[0] > self.plug_in_oracle.lookup(i, new_centroid)[0]:
                tc_jump += self.plug_in_oracle.lookup(i, new_centroid)[0] - self.plug_in_oracle.lookup(cluster_id, i)[1]
        return tc_jump
