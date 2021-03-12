from math import inf
from  random import seed, choice, sample

class CLARANS:
    def __init__(self, oracle, n, k, num_local=None, max_neighbour=None):
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
            self.max_neighbour = int(0.2 * self.k * ( self.n - self.k ) )
        self.iterate()

    def iterate(self):
        list_k = list(range(self.k))
        set_n = set(range(self.n))
        for i in range(self.num_local):
            self._assign()
            j = 1
            while j <= self.max_neighbour:
                loc = choice(list_k)
                new_centroid = choice(list(set_n.difference(set(self.local_centroids))))
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
        cost = 0
        for i in range(self.n):
            if i in self.local_centroids:
                self._clusters[i] = i
                continue
            dist = 2
            centroid = -1
            for j in range(self.k):
                if self.oracle(i, self.local_centroids[j]) < dist:
                    dist = self.oracle(i, self.local_centroids[j])
                    centroid = self.local_centroids[j]
            self._clusters[i] = centroid
            cost += dist
        for index,cen in enumerate(self.local_centroids):
            self._reverse_centroids[cen] = index
        self.local_cost = cost
        if self.local_cost < self.min_cost:
            self.min_cost = self.local_cost
            self.centroids = self.local_centroids

    def _replace(self, old_centroid, new_centroid):
        # print(old_centroid, new_centroid, self.local_centroids, self._reverse_centroids)
        for index,centroid in enumerate(self.local_centroids):
            if centroid == old_centroid:
                self.local_centroids[index] = new_centroid
                self._reverse_centroids[new_centroid] = index
                del self._reverse_centroids[centroid]
        self._clusters[new_centroid] = new_centroid
        for i in range(self.n):
            if i in self.local_centroids:
                continue
            cluster_id = self._clusters[i]
            if cluster_id == old_centroid:
                dist = 2
                _id = -1
                for j in range(self.k):
                    if dist > self.oracle(self.local_centroids[j], i):
                        dist = self.oracle(self.local_centroids[j], i)
                        _id = self.local_centroids[j]
                self._clusters[i] = _id
            elif self.oracle(cluster_id, i) >= self.oracle(i, new_centroid):
                self._clusters[i] = new_centroid

    def _calculate_diff(self, centroid_index, new_centroid):
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
                    dist = min(dist, self.oracle(self.local_centroids[j], i))
                dist = min(dist, self.oracle(new_centroid, i))
                tc_jump += dist - self.oracle(i, old_centroid)
            elif self.oracle(cluster_id, i) > self.oracle(i, new_centroid):
                tc_jump += self.oracle(i, new_centroid) - self.oracle(cluster_id, i)
        return tc_jump
