import igraph
import pickle

class StandardGraphGenerator:
    def __init__(self, n):
        """
        :param n: number of nodes
        """
        # print(igraph.__version__)
        self.n = n
        self.graph = None
        self.is_weighted = False

    def get_geometric_graph(self, radius, torus=False):
        """
        :param radius: picks the points randomly on a unit square and choose those points which are within
        the range of radius
        :param torus: you can use a square or a torus object to pick points from
        """
        self.graph = igraph.Graph.GRG(self.n, radius, torus)
        # return igraph.Graph(self.n, radius, torus)

    def get_barabasi(self, connectivity):
        """
        :param connectivity: number of connectivity for each node
        """
        self.graph = igraph.Graph.Barabasi(n=self.n, m=connectivity)
        # return igraph.Graph.Barabasi(n=self.n, m=m)

    def get_erods_renyi(self, prob):
        """
        :param prob: probability of an object being a part of the graph
        :param m: which could be a number of edegs to be in final graph(should be missing if p is not None)
        """
        self.graph = igraph.Graph.Erdos_Renyi(self.n, p=prob, directed=False, loops=False)

    def get_forrest_fire(self, fw_prob):
        """
        :param fw_prob: probability of an object being a part of the graph
        :param m: which could be a number of edegs to be in final graph(should be missing if p is not None)
        :param ambs: no of ambassadors in each step
        """
        self.graph = igraph.Graph.Forest_Fire(self.n, fw_prob, bw_factor=0.0, ambs=1, directed=False)

    def graph_visualize(self):
        if self.graph is None:
            print("No graph to print. Graph object is None.")
            return
        g = self.graph
        i = g.community_infomap()
        colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]*20
        g.vs['color'] = [None]
        for clid, cluster in enumerate(i):
            for member in cluster:
                g.vs[member]['color'] = colors[clid]
        g.vs['frame_width'] = 0
        igraph.plot(g)

    def assign_weights(self, weights):
        """
        :param weights: weight vector for assigning edge weights to graph object
        """
        assert len(weights) == self.graph.g.ecount()
        try:
            self.graph.es['weight'] = weights
            self.is_weighted = True
        except:
            self.is_weighted = False

    def check_one_connected(self):
        assert self.graph is not None
        return self.graph.is_connected()

    def check_and_write(self, file_name, write=True):
        if self.check_one_connected():
            print(self.graph.ecount())
            if write:
                pickle.dump(self.graph, open(file_name, "wb"))
