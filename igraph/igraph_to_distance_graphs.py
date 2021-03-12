import pickle
import numpy as np
from random import sample

class iGraph2DistGraph:
    def __init__(self):
        self.out = None
        self.distances = {}
        self.normal = None
        self.uniform = None
        self.zipf = None

        self.write_status = False

    def set_graph(self, pickled_graph):
        assert pickled_graph is not None
        self.igraph = pickle.load(open(pickled_graph, "rb"))

    def get_distances(self, weights):
        for i in range(weights.shape[0]):
            x = np.linalg.norm(weights - weights[i, :], axis=1).reshape((1, weights.shape[0]))
            if self.out is None:
                self.out = x
            else:
                self.out = np.vstack((self.out, x))
        if np.max(self.out) > 1:
            self.out /= np.max(self.out)

    def get_graph_dict(self):
        self.distances = {}
        for each in self.igraph.es:
            i, j = each.tuple
            self.distances[(i, j)] = self.out[i, j]

    def add_normal_weights(self, mu, sigma, dim, file_name):
        if self.normal is None:
            weights = np.random.normal(mu, sigma, (self.igraph.vcount(), dim))
            self.get_distances(weights)
            self.normal = self.out
            self.pickle_distance_dict(self.out, 'normal' +str(self.igraph.vcount()) + '.pkl')
        else:
            self.out = self.normal
        self.get_graph_dict()
        self.out = None
        self.pickle_distance_dict(self.distances, 'normal_distances_' + file_name + '.pkl')

    def add_uniform_weights(self, dim, file_name):
        if self.uniform is None:
            weights = np.random.uniform(size=(self.igraph.vcount(), dim))
            self.get_distances(weights)
            self.uniform = self.out
            self.pickle_distance_dict(self.out, 'uniform' + str(self.igraph.vcount()) + '.pkl')
        else:
            self.out = self.uniform
        self.get_graph_dict()
        self.out = None
        self.pickle_distance_dict(self.distances, 'uniform_distances_' + file_name + '.pkl')

    def add_zipf_weights(self, a, dim, file_name):
        if self.zipf is None:
            weights = np.random.zipf(a, size=(self.igraph.vcount(), dim))
            self.get_distances(weights)
            self.zipf = self.out
            self.pickle_distance_dict(self.out, 'zipf' + str(self.igraph.vcount()) + '.pkl')
        else:
            self.out = self.zipf
        self.get_graph_dict()
        self.out = None
        self.pickle_distance_dict(self.distances, 'zipf_distances_' + file_name + '.pkl')

    def pickle_distance_dict(self, distance, file_name):
        if self.write_status:
            pickle.dump(distance, open(file_name, 'wb'))

    def sample_and_add_weights(self, points, order_val, file_name):
        self.out = None
        weights = sample(list(points), order_val)
        self.get_distances(np.array(weights))

        self.zipf = self.out
        # file_name.split('_')[:2]
        self.pickle_distance_dict(self.out, "_".join(file_name.split('_')[:2]) + str(self.igraph.vcount()) + '.pkl')
        self.get_graph_dict()
        self.pickle_distance_dict(self.distances, file_name)
