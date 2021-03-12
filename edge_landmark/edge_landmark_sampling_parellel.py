import random
import sys
import os

sys.path.append('..')
from dijkstra import Dijkstra
from helper import _get_dbl_level_dict


class EdgeLandMark:
    def __init__(self, G, n, nodes, Sampling=True):
        """
        :param G: Input Graph in the form of  dictionary; G(E: <edge_weight>)
        :param n: Number of samples you need from the unknown edges for the algorithm
        :param nodes: number of nodes in the system
        """
        self.n = n
        self.G = G
        self.known = list(G.keys())
        self.nodes = list(range(nodes))
        self.sampled = []
        self.greedyK = None
        self.dijkPaths = None
        self.landmarks = None
        self.Sampling = Sampling
        self.d = None

        if not Sampling:
            for u in range(len(self.nodes)):
                for v in range(u + 1, len(self.nodes)):
                    if (u, v) in self.known:
                        pass
                    elif (v, u) in self.known:
                        pass
                    else:
                        self.sampled.append((u, v))
        else:
            while len(self.sampled) <= self.n:
                [u, v] = random.choices(self.nodes, k=2)
                if u > v:
                    u, v = v, u
                if u == v:
                    pass
                if (u, v) in self.known:
                    pass
                elif (v, u) in self.known:
                    pass
                else:
                    self.sampled.append((u, v))

    def find_paths(self):
        self.d = Dijkstra(_get_dbl_level_dict(self.G), len(self.nodes))
        self.dijkPaths = {}
        for each in self.sampled:
            for e in each:
                if self.dijkPaths.get(e, -1) == -1:
                    self.dijkPaths[e] = self.d.shortest_path(self.nodes, e)

    def greedy_sampling(self, landmarks):
        """
        This function greedily selects the edges from the known edges for our greedy algorithm
        :param landmarks: number of sample edges needed from the known edge set
        :return:
        """

        file_writer = open(os.path.join("elm-results", 'intermediate.txt'), 'a+')
        file_writer.write("{}\n".format("*" * 50))
        file_writer.write(
            "Value of Graph: {}\nValue of k: {}\nSampling State: {}\n{}\n".format(len(self.nodes), landmarks,
                                                                                  self.Sampling, "*" * 50))
        file_writer.flush()
        self.landmarks = landmarks
        previous = -1
        out = [(-1, -1)]
        if self.landmarks < 1:
            pass
        for knownEdge in self.known:
            total = 0
            for edge in self.sampled:
                total += max(0,
                             self.G[knownEdge] - self.dijkPaths[edge[0]][knownEdge[0]][0] -
                             self.dijkPaths[edge[1]][knownEdge[1]][0],
                             self.G[knownEdge] - self.dijkPaths[edge[0]][knownEdge[1]][0] -
                             self.dijkPaths[edge[1]][knownEdge[0]][0])
            if total > previous:
                previous = total
                out[-1] = knownEdge
        # net_total += previous
        if self.dijkPaths.get(out[-1][0], -1) == -1:
            self.dijkPaths[out[-1][0]] = self.d.shortest_path(self.nodes, out[-1][0])
        if self.dijkPaths.get(out[-1][1], -1) == -1:
            self.dijkPaths[out[-1][1]] = self.d.shortest_path(self.nodes, out[-1][1])
        net_total = self.get_unknown_edge_estimates(out)
        print("Iteration 1: " + str(net_total))
        file_writer.write("Iteration 1: " + str(net_total) + "\n")
        file_writer.flush()
        self.known.remove(out[-1])
        representative = [out[-1]] * len(self.sampled)
        # import pdb
        # pdb.set_trace()
        for i in range(self.landmarks - 1):
            out.append((-1, -1))
            previous = -1
            for knownEdge in self.known:
                total = 0
                for index, edge in enumerate(self.sampled):
                    cost = max(0,
                               self.G[knownEdge] - self.dijkPaths[edge[0]][knownEdge[0]][0] -
                               self.dijkPaths[edge[1]][knownEdge[1]][0],
                               self.G[knownEdge] - self.dijkPaths[edge[0]][knownEdge[1]][0] -
                               self.dijkPaths[edge[1]][knownEdge[0]][0])
                    repr = representative[index]
                    cost_current = max(0,
                                       self.G[repr] - self.dijkPaths[edge[0]][repr[0]][0] -
                                       self.dijkPaths[edge[1]][repr[1]][0],
                                       self.G[repr] - self.dijkPaths[edge[0]][repr[1]][0] -
                                       self.dijkPaths[edge[1]][repr[0]][0])
                    if cost > cost_current:
                        total += cost - cost_current
                if total > previous:
                    previous = total
                    out[-1] = knownEdge
            # net_total += previous
            if self.dijkPaths.get(out[-1][0], -1) == -1:
                self.dijkPaths[out[-1][0]] = self.d.shortest_path(self.nodes, out[-1][0])
            if self.dijkPaths.get(out[-1][1], -1) == -1:
                self.dijkPaths[out[-1][1]] = self.d.shortest_path(self.nodes, out[-1][1])
            net_total = self.get_unknown_edge_estimates(out)
            print("Iteration " + str(i + 2) + ": " + str(net_total))
            file_writer.write("Iteration " + str(i + 2) + ": " + str(net_total) + "\n")
            file_writer.flush()
            # print(previous)
            for index, edge in enumerate(self.sampled):
                cost = max(0,
                           self.G[out[-1]] - self.dijkPaths[edge[0]][out[-1][0]][0] -
                           self.dijkPaths[edge[1]][out[-1][1]][0],
                           self.G[out[-1]] - self.dijkPaths[edge[0]][out[-1][1]][0] -
                           self.dijkPaths[edge[1]][out[-1][0]][0])
                repr = representative[index]
                cost_current = max(0,
                                   self.G[repr] - self.dijkPaths[edge[0]][repr[0]][0] -
                                   self.dijkPaths[edge[1]][repr[1]][0],
                                   self.G[repr] - self.dijkPaths[edge[0]][repr[1]][0] -
                                   self.dijkPaths[edge[1]][repr[0]][0])
                if cost > cost_current:
                    representative[index] = out[-1]
            self.known.remove(out[-1])
        self.greedyK = out

        d = Dijkstra(_get_dbl_level_dict(self.G), len(self.nodes))
        for each in self.greedyK:
            for e in each:
                if self.dijkPaths.get(e, -1) == -1:
                    self.dijkPaths[e] = d.shortest_path(self.nodes, e)

        new_total = self.get_unknown_edge_estimates(self.greedyK)
        print(new_total)
        for i in range(len(self.nodes)):
            if i % 100 == 0:
                print(i)
            for j in range(i + 1, len(self.nodes)):
                edge = (i, j)
                some = 0
                if self.G.get((i, j), -1) == -1:
                    for repr in self.greedyK:
                        some = max(some,
                                   self.G[repr] - self.dijkPaths[repr[0]][edge[0]][0] -
                                   self.dijkPaths[repr[1]][edge[1]][0],
                                   self.G[repr] - self.dijkPaths[repr[1]][edge[0]][0] -
                                   self.dijkPaths[repr[0]][edge[1]][0])
                    new_total += some
        print("\nEdge Landmark Sum: {} for k(Land Marks):{} ".format(new_total, self.landmarks))
        file_writer.write("=" * 50 + "\n")
        file_writer.flush()
        file_writer.close()

        return new_total

    def get_unknown_edge_estimates(self, greedyK_local):
        new_total = 0
        for i in range(len(self.nodes)):
            if i % 100 == 0:
                print(i)
            for j in range(i + 1, len(self.nodes)):
                edge = (i, j)
                some = 0
                if self.G.get((i, j), -1) == -1:
                    for repr in greedyK_local:
                        some = max(some,
                                   self.G[repr] - self.dijkPaths[repr[0]][edge[0]][0] -
                                   self.dijkPaths[repr[1]][edge[1]][0],
                                   self.G[repr] - self.dijkPaths[repr[1]][edge[0]][0] -
                                   self.dijkPaths[repr[0]][edge[1]][0])
                    new_total += some
        return new_total
