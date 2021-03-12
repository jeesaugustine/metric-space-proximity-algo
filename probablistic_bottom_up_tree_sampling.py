import operator
from node import Node
from random import random

class BottomUPTree:
    def __init__(self, dist_dictionary, order):
        self.dist_dictionary = dist_dictionary
        self.sorted_dist_dictionary = sorted(self.dist_dictionary.items(), key=operator.itemgetter(1))
        self.order = order
        self.make_tree()
        self.build_tree()

    def _cluster_merge(self, v1, v2, cluster_map, cluster_index, cluster):
        cluster_map[cluster_index] = [] + cluster_map[cluster[v1]] + cluster_map[cluster[v2]]
        del cluster_map[cluster[v1]]
        del cluster_map[cluster[v2]]
        for index in cluster_map[cluster_index]:
            cluster[index] = cluster_index

    def make_tree(self, threshold = 3, prob_threshold = 0.8):
        cluster = list(range(self.order))
        cluster_map = dict(zip(range(self.order), [[i] for i in range(self.order)]))
        cluster_index = self.order
        edges = []
        flip = False
        level = 0
        levels_since_flip = 0
        while len(cluster_map) > 1:
            level += 1
            if not flip:
                levels_since_flip += 1
            else:
                levels_since_flip = 0
            i = 0
            cycle_start = cluster_index
            initial_clusters = set(list(cluster_map.keys()))
            while i < len(self.sorted_dist_dictionary):
                if flip:
                    index_i = len(self.sorted_dist_dictionary) - 1 - i
                else:
                    index_i = i
                key, value = self.sorted_dist_dictionary[index_i]
                v1, v2 = key
                if cluster[v1] < cycle_start and cluster[v2] < cycle_start and cluster[v1] != cluster[v2]:
                    self._cluster_merge(v1, v2, cluster_map, cluster_index, cluster)
                    edges.append(((min(v1, v2), max(v1, v2)), self.dist_dictionary[key]))
                    cluster_index += 1
                if cluster[v1] == cluster[v2]:
                    self.sorted_dist_dictionary.pop(index_i)
                else:
                    i += 1
            final_clusters = set(list(cluster_map.keys()))
            unmerged = initial_clusters.intersection(final_clusters)
            i = 0
            while i < len(self.sorted_dist_dictionary):
                if flip:
                    index_i = len(self.sorted_dist_dictionary) - 1 - i
                else:
                    index_i = i
                key, value = self.sorted_dist_dictionary[index_i]
                v1, v2 = key
                if cluster[v1] in unmerged or cluster[v2] in unmerged:
                    v1_clus = cluster[v1]
                    v2_clus = cluster[v2]
                    self._cluster_merge(v1, v2, cluster_map, cluster_index, cluster)
                    edges.append(((min(v1, v2), max(v1, v2)), self.dist_dictionary[key]))
                    cluster_index += 1
                    if v1_clus in unmerged:
                        unmerged.remove(v1_clus)
                    else:
                        unmerged.remove(v2_clus)
                if cluster[v1] == cluster[v2]:
                    self.sorted_dist_dictionary.pop(index_i)
                else:
                    i += 1
                if len(unmerged) == 0:
                    break
            #assert len(unmerged) == 0
            if len(cluster_map) <= threshold: 
                flip = True
            elif flip:
                flip = False
            elif random() * levels_since_flip > prob_threshold:
                flip = True
        self.root = edges[-1][0][0]
        self.tree_dict = dict(edges)
        self.levels = level

    def build_tree(self):
        # index, distance, parent, children
        multi_dict = {}
        for key, value in self.tree_dict.items():
            n1, n2 = key
            if n1 not in multi_dict:
                multi_dict[n1] = {}
            if n2 not in multi_dict:
                multi_dict[n2] = {}
            multi_dict[n1][n2] = value
            multi_dict[n2][n1] = value
        self.tree = self._build_tree(multi_dict, self.root, set([self.root]))

    def _build_tree(self, multi_dict, node_id, visited):
        child_nodes = set(multi_dict[node_id].keys()).difference(visited)
        visited |= child_nodes
        children = []
        me = Node(node_id, 0, None, None)
        for c in child_nodes:
            child = self._build_tree(multi_dict, c, visited)
            child.parent = me
            child.distance = multi_dict[min(node_id, c)][max(node_id, c)]
            children.append(child)
        me.children = children
        return me
