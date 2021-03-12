import random
import heapq


class Tree:
    def __init__(self, parent, left, right, _id):
        self.parent = parent
        self.left = left
        self.right = right
        self.id = _id


class kNN_Rp:
    def __init__(self, n, k, oracle, seed=59):
        self.n = n
        self.k = k
        self.oracle = oracle
        random.seed(seed)
        self.NHA = dict(zip(list(range(n)), [[] for i in range(n)]))
        self.PR = [-1] * self.n
        self.nn_queries = []
        self.ref = [None] * self.n

    def _put_nha(self, i, j, dist):
        # Storing negative to keep min heap and use
        # heapq method from library
        item = (-dist, j)
        if item in self.NHA[i]:
            return
        if len(self.NHA[i]) < self.k:
            heapq.heappush(self.NHA[i], item)
        elif (- self.NHA[i][0][0]) > dist:
            heapq.heappushpop(self.NHA[i], item)
        return

    def _update_nha(self, i, j):
        dist = self.oracle(i, j)
        self._put_nha(i, j, dist)
        self._put_nha(j, i, dist)
        return dist

    def _get_farthest_points(self, points):
        if len(points) == 1:
            return tuple(points)
        random.shuffle(points)
        min_dist = 2.  # 2 is infinity for us
        pair = None
        for i in range(int(len(points) / 2)):
            if i + 1 < len(points):
                dist = self._update_nha(points[i], points[i + 1])
                if dist < min_dist:
                    pair = (points[i], points[i + 1])
        return pair

    def _assign(self, parents, points):
        points_1 = []
        points_2 = []
        for p in points:
            dist_1 = self._update_nha(p, parents[0])
            dist_2 = self._update_nha(p, parents[1])
            if dist_1 < dist_2:
                points_1.append(p)
                if dist_1 > self.PR[parents[0]]:
                    self.PR[parents[0]] = dist_1
            else:
                points_2.append(p)
                if dist_2 > self.PR[parents[1]]:
                    self.PR[parents[1]] = dist_2
        return (points_1, points_2)

    def _form_index(self, node, points):
        if points is None or len(points) == 0:
            return None
        child_points = self._get_farthest_points(points)
        new_left_node = Tree(node, None, None, child_points[0])
        self.ref[child_points[0]] = new_left_node
        node.left = new_left_node
        points.remove(child_points[0])
        if len(child_points) == 2:
            new_right_node = Tree(node, None, None, child_points[1])
            self.ref[child_points[1]] = new_right_node
            node.right = new_right_node
            points.remove(child_points[1])
        if len(points) > 0:
            partitioned_points = self._assign(child_points, points)
            if len(partitioned_points[0]) > 0:
                self._form_index(new_left_node, partitioned_points[0])
            if len(partitioned_points[1]) > 0:
                self._form_index(new_right_node, partitioned_points[1])

    def _make_n_queries_heap(self, root):
        if root != None:
            if root.id > 0:
                if len(self.NHA[root.id]) < self.k:
                    self.nn_queries.append([2, root.id])
                else:
                    self.nn_queries.append([self.NHA[root.id][self.k - 1][0], root.id])
            self._make_n_queries_heap(root.left)
            self._make_n_queries_heap(root.right)

    def _get_next(self, path_parent, path_pparent):
        if path_parent is None or path_pparent is None:
            return None
        if path_pparent.left == path_parent:
            return path_pparent.right
        else:
            return path_pparent.left

    def _update(self, cur, node):
        if not node:
            return
        dist = self._update_nha(cur.id, node.id)
        # -self.NHA[node.id][0] because of min heap.
        if len(self.NHA[node.id]) < self.k or self.PR[node.id] + (-self.NHA[node.id][0][0]) > dist:
            self._update(cur, node.left)
            self._update(cur, node.right)

    def knn_queries(self):
        root = Tree(None, None, None, -1)
        self._form_index(root, list(range(self.n)))
        self.index = root
        self._make_n_queries_heap(root)
        heapq.heapify(self.nn_queries)
        while len(self.nn_queries) > 0:
            # Heap <- (distance, node) 
            top = heapq.heappop(self.nn_queries)
            cur_node = self.ref[top[1]]
            path_parent = self.ref[top[1]].parent
            path_pparent = path_parent.parent
            next_node = self._get_next(path_parent, path_pparent)
            while next_node:
                if next_node:
                    self._update(cur_node, next_node.left)
                    self._update(cur_node, next_node.right)
                path_parent = path_pparent
                path_pparent = path_parent.parent
                next_node = self._get_next(path_parent, path_pparent)
