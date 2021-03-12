from math import inf
from  random import choices, sample

class PAM:
    def __init__(self, oracle, n, k, centroids=None):
        self.n = n
        self.k = k
        self.oracle = oracle
        self.centroids = centroids
        self.clusters = {}
        self.reverse_centroids = {}
        self._assign()
        self.iterate()

    def _assign(self):
        if self.centroids is None:
            self.centroids = sample(list(range(self.n)), self.k)
        for i in range(self.n):
            if i in self.centroids:
                self.clusters[i] = i
                continue
            dist = 2
            centroid = -1
            for j in range(self.k):
                if self.oracle(i, self.centroids[j]) < dist:
                    dist = self.oracle(i, self.centroids[j])
                    centroid = self.centroids[j]
            self.clusters[i] = centroid
        for index,cen in enumerate(self.centroids):
            self.reverse_centroids[cen] = index

    def iterate(self):
        finished = False
        while not finished:
            finished = True
            indices = set(range(self.n)).difference(set(self.centroids))
            tc_jump_least = inf
            old_cluster = -1
            new_cluster = -1
            for i in indices:
                tc_jump = self._calculate_diff(self.reverse_centroids[self.clusters[i]], i)
                if tc_jump < tc_jump_least:
                    tc_jump_least = tc_jump
                    old_cluster = self.clusters[i]
                    new_cluster = i
            if tc_jump_least < 0:
                self._replace(old_cluster, new_cluster)
                finished = False

    def _replace(self, old_centroid, new_centroid):
        for index,centroid in enumerate(self.centroids):
            if centroid == old_centroid:
                self.centroids[index] = new_centroid
                self.reverse_centroids[new_centroid] = index
                del self.reverse_centroids[centroid]
        self.clusters[new_centroid] = new_centroid
        for i in range(self.n):
            if i in self.centroids:
                continue
            cluster_id = self.clusters[i]
            if cluster_id == old_centroid:
                dist = 2
                id = -1
                for j in range(self.k):
                    if dist > self.oracle(self.centroids[j], i):
                        dist = self.oracle(self.centroids[j], i)
                        id = self.centroids[j]
                self.clusters[i] = id
            elif self.oracle(cluster_id, i) >= self.oracle(i, new_centroid):
                self.clusters[i] = new_centroid

    def _calculate_diff(self, centroid_index, new_centroid):
        tc_jump = 0
        old_centroid = self.centroids[centroid_index]
        for i in range(self.n):
            if i != old_centroid and i in self.centroids:
                continue
            cluster_id = self.clusters[i]
            if cluster_id == old_centroid:
                dist = 2
                for j in range(self.k):
                    if j == centroid_index:
                        continue
                    dist = min(dist, self.oracle(self.centroids[j], i))
                dist = min(dist, self.oracle(new_centroid, i))
                tc_jump += dist - self.oracle(i, old_centroid)
            elif self.oracle(cluster_id, i) > self.oracle(i, new_centroid):
                tc_jump += self.oracle(i, new_centroid) - self.oracle(cluster_id, i)
        return tc_jump
