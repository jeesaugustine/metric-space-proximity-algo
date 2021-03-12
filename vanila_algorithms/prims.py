class Prims:
    def __init__(self, order, oracle):
        self.oracle = oracle
        self.order = order
        self.mst_dict = dict()
        self.mst_path_length = None


    def mst(self, source, nodes=None):
        if nodes is None:
            nodes = list(range(self.order))
        mst_dict = dict()
        set_nodes = set(nodes)
        # (distance, cur_node, parent)
        heap = [[0, source, source]]
        ptr = {}
        total = 0
        self.dist_dict = []
        for edge_i in range(self.order):
            self.dist_dict.append([])
            for edge_j in range(self.order):
                if edge_j == edge_i:
                    self.dist_dict[edge_i].append(0)
                elif edge_i > edge_j:
                    self.dist_dict[edge_i].append(self.dist_dict[edge_j][edge_i])
                else:
                    self.dist_dict[edge_i].append(self.oracle(edge_i, edge_j))
        while heap:
            top = self._heap_pop(heap, ptr)
            set_nodes.remove(top[1])
            cur_length = top[0]
            if top[2] != top[1]:
                mst_dict[(top[2], top[1])] = cur_length
            total += cur_length
            for n in range(self.order):
                if n not in set_nodes:
                    continue
                if n in ptr and heap[ptr[n]][0] > self.dist_dict[top[1]][n]:
                    self._update(n, self.dist_dict[top[1]][n], top[1], ptr, heap)
                elif n not in ptr:
                    self._insert(n, self.dist_dict[top[1]][n], top[1], ptr, heap)
        print("Vanila Prims Length: ", total)
        self.mst_dict = mst_dict
        self.mst_path_length = total

    def _swap(self, heap, index1, index2, ptr):
        temp = heap[index1]
        heap[index1] = heap[index2]
        heap[index2] = temp
        ptr[heap[index1][1]] = index1
        ptr[heap[index2][1]] = index2

    def _heap_pop(self, heap, ptr):
        if len(heap) == 1:
            return heap.pop()
        last = heap.pop()
        top = heap[0]
        index = 0
        heap[0] = last
        ptr[last[1]] = 0
        distance = last[0]
        while index < len(heap):
            left_ptr = 2 * index + 1
            right_ptr = left_ptr + 1
            if left_ptr < len(heap) and right_ptr < len(heap) and min(heap[left_ptr][0], heap[right_ptr][0]) < distance:
                if heap[left_ptr][0] < heap[right_ptr][0]:
                    self._swap(heap, left_ptr, index, ptr)
                else:
                    self._swap(heap, right_ptr, index, ptr)
            elif left_ptr < len(heap) and heap[left_ptr][0] < distance:
                self._swap(heap, left_ptr, index, ptr)
            else:
                return top
            index = ptr[last[1]]
        del ptr[top[1]]
        return top

    def _insert(self, n, distance, parent, ptr, heap):
        heap.append([distance, n, parent])
        ptr[n] = len(heap) - 1
        self._put_into_position(ptr[n], heap, ptr)

    def _update(self, n, distance, parent, ptr, heap):
        heap[ptr[n]][0] = distance
        heap[ptr[n]][2] = parent
        self._put_into_position(ptr[n], heap, ptr)

    def _put_into_position(self, loc, heap, ptr):
        while loc != 0:
            parent = int((loc-1)/2)
            if heap[parent][0] > heap[loc][0]:
                self._swap(heap, parent, loc, ptr)
                loc = parent
            else:
                return