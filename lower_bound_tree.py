class LBTree:
    def __init__(self, root, lb_matrix, subtrees=True):
        self.root = root
        self.lb_matrix = lb_matrix
        self.root.path_length = 0
        self.root.max = 0
        self._update_matrix(self.root, subtrees)

    def update_distances(self, node, max_value, path_length):
        node.path_length = node.distance + path_length
        node.max = max(node.distance, max_value)
        for c in node.children:
            self.update_distances(c, node.max, node.path_length)

    def _update_lb_btw_nodes(self, node1, node2):
        val = 2 * max(node1.max, node2.max) - node1.path_length - node2.path_length
        self.lb_matrix[node1.index][node2.index] = self.lb_matrix[node2.index][node1.index] = max(self.lb_matrix[node1.index][node2.index], val)

    def _vary_right(self, left, right):
        self._update_lb_btw_nodes(left, right)
        for c in right.children:
            self._vary_right(left, c)

    def _vary_left_right(self, left, right):
        self._vary_right(left, right)
        for c in left.children:
            self._vary_left_right(c, right)

    def _update_matrix(self, node, subtrees):
        for i in node.children:
            self.update_distances(i, 0, 0)
        children_count = len(node.children)
        for i in range(children_count):
            self._vary_right(node, node.children[i])
            for j in range(i + 1, children_count):
                self._vary_left_right(node.children[i], node.children[j])
        if not subtrees:
            return
        for i in node.children:
            i.max = 0
            i.path_length = 0
            self._update_matrix(i, subtrees)
