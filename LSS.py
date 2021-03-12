from dijkstra import Dijkstra
from helper import _get_dbl_level_dict
import random, copy
from sortedcontainers import SortedList

def select_the_new_landmark(sum_node_to_landmark):
    max_key = max(sum_node_to_landmark, key=sum_node_to_landmark.get)

    return max_key


class LSS:
    def __init__(self, g, order_val, k, vanila_oracle):
        self.G = g
        self.noNodes = order_val
        self.vertices = [i for i in range(self.noNodes)]
        self.k = k
        self.k1 = k
        self.vanila_oracle = vanila_oracle
        # self.pick_landmarks()
        self.sp = dict()
        self.nodeLandmarks = list()

        random.seed(30)
        # self.pick_random_nodelandmarks()
        self.edgeLandmarks = SortedList([], key=lambda x: x['length'])
        self.get_landmarks() # This will initilaize all the node

        self.more_node_landmarks = copy.copy(self.k_landmarks)
        self.lookup_ctr = 0
        self.lb_better_ctr = 0
        self.ub_better_ctr = 0

    def get_landmarks(self):
        assert self.k >= 1
        # self.k_landmarks = [random.choice(self.vertices)]
        self.k_landmarks = [0]
        sum_node_to_landmark = {}
        select = self.k_landmarks[-1]
        import copy
        dup_nodes = copy.copy(self.vertices)

        while len(self.k_landmarks) < self.k + 1:
            # sum_node_to_landmark = {}
            for node in range(self.noNodes):
                if node in self.k_landmarks:
                    continue
                u, v = min(select, node), max(select, node)
                if sum_node_to_landmark.get(node, -1) == -1:
                    sum_node_to_landmark[node] = 0
                if self.G.get((u, v), -1) == -1:
                    dist_val = self.vanila_oracle(u, v)
                    # self.update((u, v), dist_val)
                else:
                    dist_val = self.G[(u, v)]
                self.G[(u, v)] = dist_val
                if len(self.edgeLandmarks) < self.k1:
                    self.edgeLandmarks.add({'edge': (u, v), "length": dist_val})
                    # print("Val 53 here: {}, vals:{}".format((u, v), dist_val))
                elif self.edgeLandmarks[0]['length'] < dist_val:
                    del self.edgeLandmarks[0]
                    self.edgeLandmarks.add({'edge': (u, v), "length": dist_val})
                    # print("Val 57 here: {}, vals:{}".format((u, v), dist_val))

                sum_node_to_landmark[node] += dist_val
            select = select_the_new_landmark(sum_node_to_landmark)
            dup_nodes.remove(select)
            sum_node_to_landmark.pop(select)

            self.k_landmarks.append(select)
        del self.k_landmarks[-1]
        print("Landmarks(LSS): {}".format(self.k_landmarks))
        self.collector = dict(zip(self.k_landmarks, [1] * len(self.k_landmarks)))
        for edge_details in self.edgeLandmarks:
            a, b = edge_details["edge"]
            a, b = min(a, b), max(a, b)
            if a in self.collector:
                self.collector[a] += 1
            else:
                self.collector[a] = 1
            if b in self.collector:
                self.collector[b] += 1
            else:
                self.collector[b] = 1

        dijk_obj = Dijkstra(_get_dbl_level_dict(self.G, self.noNodes), self.noNodes)
        # print("Keys of Collector {}".format(self.collector.keys()))
        # print("Node Landmarks: {}".format(self.k_landmarks))
        for landmark in self.collector:
            if landmark in self.sp:
                continue
            self.sp[landmark] = dijk_obj.shortest_path(self.vertices, landmark)

    def pick_node_landmarks(self):
        pass

    def pick_random_nodelandmarks(self):
        self.nodeLandmarks = random.sample(self.vertices, self.k)

    def store_nodeLandmarks(self):
        dijk_obj = Dijkstra(_get_dbl_level_dict(self.G, self.noNodes), self.noNodes)
        for landmark in self.nodeLandmarks:
            if landmark in self.sp:
                continue
            self.sp[landmark] = dijk_obj.shortest_path(self.vertices, landmark)

    def pick_random_edgeLandmarks(self):
        mylist = random.sample(self.vertices, 2 * self.k)
        self.edgeLandmarks = list(zip(mylist[0::2], mylist[1::2]))
        for (u, v) in self.edgeLandmarks:
            a, b = min(u, v), max(u, v)
            val = self.vanila_oracle(a, b)
            self.G[(a, b)] = val
            # print(val)

    def store_edgeLandmarks(self):
        dijk_obj = Dijkstra(_get_dbl_level_dict(self.G, self.noNodes), self.noNodes)
        for (x, y) in self.edgeLandmarks:
            if x not in self.sp:
                self.sp[x] = dijk_obj.shortest_path(self.vertices, x)
            if y not in self.sp:
                self.sp[y] = dijk_obj.shortest_path(self.vertices, y)

    def lookup(self, x, y):
        lb = 0
        ub = 1
        if not self.is_uncalculated(x, y):
            return [self.G[(min(x, y), max(x, y))]] * 2
        for edge_details in self.edgeLandmarks:
            u, v = edge_details["edge"]
            u, v = min(u, v), max(u, v)
            if x in self.sp[u] and y in self.sp[u]:
                lb = max(lb, self.G[(u, v)] - self.sp[u][x][0] - self.sp[v][y][0],
                         self.G[(u, v)] - self.sp[u][y][0] - self.sp[v][x][0])
        for landmark in self.sp.keys():
            if x in self.sp[landmark] and y in self.sp[landmark]:
                ub = min(ub, self.sp[landmark][x][0] + self.sp[landmark][y][0])
        l_lb, l_ub = self.laesa_lookup(x, y)
        if self.lookup_ctr % 5000 == 0:
            print([lb, ub])
        self.lookup_ctr += 1

        if lb > l_lb:
            self.lb_better_ctr += 1
        if ub < l_ub:
            self.ub_better_ctr += 1

        return [max(lb, l_lb), min(ub, l_ub)]

    def laesa_lookup(self, x, y):
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

    def update(self, edge, val):
        # print("\nEdge: {}; val: {}\n\n".format(edge, val))
        u, v = min(edge), max(edge)
        self.G[(u, v)] = val

        dijk_obj = Dijkstra(_get_dbl_level_dict(self.G, self.noNodes), self.noNodes)
        new_sp_u = dijk_obj.shortest_path(self.vertices, u)
        new_sp_v = dijk_obj.shortest_path(self.vertices, v)

        self.sp[u] = new_sp_u
        self.sp[v] = new_sp_v
        for key in self.sp.keys():
            if key == u or key == v:
                continue
            if u not in self.sp[key] and v not in self.sp[key]:
                continue

            for node in self.vertices:
                if key in new_sp_u and node in new_sp_u:
                    change_len_node = min(1, new_sp_u[key][0] + new_sp_v[node][0] + val,
                                          new_sp_u[node][0] + new_sp_v[key][0] + val)
                    if node in self.sp[key]:
                        self.sp[key][node][0] = min(change_len_node, self.sp[key][node][0])
                    else:
                        self.sp[key][node] = [change_len_node]

        local_total = SortedList([], key=lambda x: x['length'])
        for each in self.more_node_landmarks:
            total = 0
            for key, val in self.sp[each].items():
                total += val[0]
            local_total.add({'node': each, "length": total})

        if u not in self.more_node_landmarks:
            total = 0
            for key, val in new_sp_u.items():
                total += val[0]
            local_total.add({'node': u, "length": total})
        if v not in self.more_node_landmarks:
            total = 0
            for key, val in new_sp_v.items():
                total += val[0]
            local_total.add({'node': v, "length": total})
        self.more_node_landmarks = list(map(lambda x: x['node'], local_total[:-2]))
        # print(local_total, self.more_node_landmarks)

        if u in self.collector:
            self.collector[u] += 2
        else:
            self.collector[u] = 2
        if v in self.collector:
            self.collector[v] += 2
        else:
            self.collector[v] = 2

        self.collector[local_total[-1]["node"]] -= 1
        self.collector[local_total[-2]["node"]] -= 1
        if self.collector[local_total[-1]["node"]] == 0:
            # print("Deleting {}".format(self.collector[local_total[-1]["node"]]))
            del self.collector[local_total[-1]["node"]]
            del self.sp[local_total[-1]["node"]]
        if self.collector[local_total[-2]["node"]] == 0:
            # print("Deleting {}".format(self.collector[local_total[-2]["node"]]))
            del self.collector[local_total[-2]["node"]]
            del self.sp[local_total[-2]["node"]]


        # Edge landmark update

        # print("Edge Landmarks: {}".format({'edge': (u, v), "length": self.G[(u, v)]}))
        self.edgeLandmarks.add({'edge': (u, v), "length": self.G[(u, v)]})
        self.collector[self.edgeLandmarks[0]["edge"][0]] -= 1
        self.collector[self.edgeLandmarks[0]["edge"][1]] -= 1
        if self.collector[self.edgeLandmarks[0]["edge"][1]] == 0:
            # print("Deleting {}".format(self.collector[self.edgeLandmarks[-1]["edge"][1]]))
            del self.collector[self.edgeLandmarks[0]["edge"][1]]
            del self.sp[self.edgeLandmarks[0]["edge"][1]]
        if self.collector[self.edgeLandmarks[0]["edge"][0]] == 0:
            # print("Deleting {}".format(self.collector[self.edgeLandmarks[-2]["edge"][1]]))
            del self.collector[self.edgeLandmarks[0]["edge"][0]]
            del self.sp[self.edgeLandmarks[0]["edge"][0]]

        # print("Deleting Edge LM{}".format(self.edgeLandmarks[-1]))
        del self.edgeLandmarks[0]
        # print((self.edgeLandmarks))


    def is_uncalculated(self, x, y):
        return not (((x, y) in self.G) or ((y, x) in self.G))
