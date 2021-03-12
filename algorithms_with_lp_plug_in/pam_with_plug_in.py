from math import inf
from  random import sample
import copy
import pprint

class PAM:
    def __init__(self, oracle, plug_in_oracle, n, k, centroids=None):
        self.n = n
        self.k = k
        self.oracle = oracle
        self.centroids = centroids
        self.clusters = {}
        self.reverse_centroids = {}
        self.plug_in_oracle = plug_in_oracle
        self._assign()
        self.iterate()

    def _assign(self):
        if self.centroids is None:
            self.centroids = sample(list(range(self.n)), self.k)
        count = 0
        for i in range(self.n):
            #if i in self.centroids:
            #    self.clusters[i] = i
            #    continue
            dist = 2
            centroid = -1
            for j in range(self.k):
                if self.plug_in_oracle.is_uncalculated(i, self.centroids[j]):
                    self.plug_in_oracle.update((i, self.centroids[j]), self.oracle(i, self.centroids[j]))
                if dist > self.plug_in_oracle.lookup(i, self.centroids[j])[1]:
                    dist = self.plug_in_oracle.lookup(i, self.centroids[j])[1]
                    centroid = self.centroids[j]
                count += dist
            self.clusters[i] = centroid
        for index,cen in enumerate(self.centroids):
            self.reverse_centroids[cen] = index
        print(count)

    def iterate(self):
        finished = False
        while not finished:
            # print(self.centroids, self.reverse_centroids)
            finished = True
            indices = set(range(self.n)).difference(set(self.centroids))
            tc_jump_least = inf
            old_cluster = -1
            new_cluster = -1
            for i in indices:
                tc_jump = self._calculate_diff_potential(self.reverse_centroids[self.clusters[i]], i)
                if tc_jump < tc_jump_least:
                    #print(tc_jump, i)
                    tc_jump = self._calculate_diff(self.reverse_centroids[self.clusters[i]], i)
                    if tc_jump < tc_jump_least:
                        tc_jump_least = tc_jump
                        old_cluster = self.clusters[i]
                        new_cluster = i
            if tc_jump_least < 0:
                self._replace(old_cluster, new_cluster)
                finished = False

    def _replace(self, old_centroid, new_centroid):
        self.centroids[self.reverse_centroids[old_centroid]] = new_centroid
        self.reverse_centroids[new_centroid] = self.reverse_centroids[old_centroid]
        del self.reverse_centroids[old_centroid]
        self.clusters[new_centroid] = new_centroid
        for i in range(self.n):
            #if i in self.centroids:
            #    continue
            cluster_id = self.clusters[i]
            if self.plug_in_oracle.is_uncalculated(i, new_centroid):
                self.plug_in_oracle.update((i, new_centroid), self.oracle(i, new_centroid))
            if cluster_id == old_centroid:
                dist = 2
                id = -1
                for j in range(self.k):
                    if dist > self.plug_in_oracle.lookup(self.centroids[j], i)[1]:
                        dist = self.plug_in_oracle.lookup(self.centroids[j], i)[1]
                        id = self.centroids[j]
                self.clusters[i] = id
            elif self.plug_in_oracle.lookup(cluster_id, i)[1] > self.plug_in_oracle.lookup(i, new_centroid)[1]:
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
                    dist = min(dist, self.plug_in_oracle.lookup(self.centroids[j], i)[1])
                dist = min(dist, self.plug_in_oracle.lookup(new_centroid, i)[1])
                if self.plug_in_oracle.is_uncalculated(new_centroid, i) and self.plug_in_oracle.lookup(new_centroid, i)[0] < dist:
                    dist_actual = self.oracle(i, new_centroid)
                    self.plug_in_oracle.update((i, new_centroid), dist_actual)
                dist = min(dist, self.plug_in_oracle.lookup(new_centroid, i)[0])
                tc_jump += dist - self.plug_in_oracle.lookup(i, old_centroid)[1]
            elif self.plug_in_oracle.lookup(cluster_id, i)[1] > self.plug_in_oracle.lookup(i, new_centroid)[0]:
                if self.plug_in_oracle.is_uncalculated(new_centroid, i):
                    dist_actual = self.oracle(i, new_centroid)
                    self.plug_in_oracle.update((i, new_centroid), dist_actual)
                if self.plug_in_oracle.lookup(new_centroid, i)[0] < self.plug_in_oracle.lookup(cluster_id, i)[1]:
                    tc_jump += self.plug_in_oracle.lookup(i, new_centroid)[1] - self.plug_in_oracle.lookup(cluster_id, i)[1]
        return tc_jump

    def _calculate_diff_potential(self, centroid_index, new_centroid):
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
                    dist = min(dist, self.plug_in_oracle.lookup(self.centroids[j], i)[1])
                dist = min(dist, self.plug_in_oracle.lookup(new_centroid, i)[0])
                tc_jump += dist - self.plug_in_oracle.lookup(i, old_centroid)[1]
            elif self.plug_in_oracle.lookup(cluster_id, i)[0] > self.plug_in_oracle.lookup(i, new_centroid)[0]:
                tc_jump += self.plug_in_oracle.lookup(i, new_centroid)[0] - self.plug_in_oracle.lookup(cluster_id, i)[1]
        return tc_jump
