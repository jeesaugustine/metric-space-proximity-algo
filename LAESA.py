import random
import time

def select_the_new_landmark(sum_node_to_landmark):
    max_key = max(sum_node_to_landmark, key=sum_node_to_landmark.get)

    return max_key


class NodeLandMarkRandom:
    def __init__(self, g, k, oracle, order_val):
        self.G = g
        self.k = k            # Number of Landmarks to be chosen
        self.oracle = oracle  # The Oracle chosen for the
        # self.noNodes = len(set([node for nodes in list(self.G.keys()) for node in nodes]))
        self.noNodes = order_val
        self.vertices = [i for i in range(self.noNodes)]
        self.adj_list = {}
        self.seed = 30
        # self.set_seed()
        self.k_landmarks = None
        self.time2prime = 0
        self.get_landmarks()

    def get_landmarks(self):
        assert self.k >= 1
        # self.k_landmarks = [random.choice(self.vertices)]
        sum_node_to_landmark = {}

        start = time.time()
        self.k_landmarks = [0]
        select = self.k_landmarks[-1]
        import copy
        # dup_nodes = copy.copy(self.vertices)
        # dup_nodes.remove(select)

        while len(self.k_landmarks) < self.k + 1:
            # sum_node_to_landmark = {}
            for node in range(self.noNodes):
                if node in self.k_landmarks:
                    continue
                u, v = min(select, node), max(select, node)
                if sum_node_to_landmark.get(node, -1) == -1:
                    sum_node_to_landmark[node] = 0
                if self.G.get((u, v), -1) == -1:
                    dist_val = self.oracle(u, v)
                    self.update((u, v), dist_val)
                else:
                    dist_val = self.G[(u, v)]
                sum_node_to_landmark[node] += dist_val
            select = select_the_new_landmark(sum_node_to_landmark)
            # dup_nodes.remove(select)
            sum_node_to_landmark.pop(select)

            self.k_landmarks.append(select)
        del self.k_landmarks[-1]
        self.time2prime = time.time() - start

        print("Landmarks(NLM): {}".format(self.k_landmarks))

    def lookup(self, x, y):
        if x == y:
            return [0, 0]
        u, v = min(x, y), max(x, y)
        if self.G.get((u, v), -1) > -1:
            return [self.G[(u, v)], self.G[(u, v)]]
        ub = 1
        lb = 0
        for landmark in self.k_landmarks:
            u1, v1 = min(landmark, u), max(landmark, u)
            u2, v2 = min(landmark, v), max(landmark, v)

            cur_ub = self.G[(u1, v1)] + self.G[(u2, v2)]
            if cur_ub < ub:
                ub = cur_ub

            cur_lb = abs(self.G[(u1, v1)] - self.G[(u2, v2)])
            if cur_lb > lb:
                lb = cur_lb

        return [lb, ub]

    def set_seed(self):
        random.seed(self.seed)

    def update(self, edge, val):
        u, v = min(edge), max(edge)
        self.G[(u, v)] = val

    def is_uncalculated(self, x, y):
        return (x != y) and (not (((x, y) in self.G) or ((y, x) in self.G)))
