class Prims:
    def __init__(self, order, oracle, plug_in_oracle):
        # lb, ub = plug_in_oracle(u, v)
        # actual_distance = oracle(u, v)
        self.oracle = oracle
        self.plug_in_oracle = plug_in_oracle
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
        ptr = {source: 0}
        total = 0
        self.dist_dict = []
        for i in range(self.order):
            if i % 1000 == 0:
                print("Number of Edges found by primes: {}".format(i))
            top = self._heap_pop(heap, ptr)
            if top[2] != top[1]:
                mst_dict[(top[2], top[1])] = top[0]
                total += top[0]
            set_nodes.remove(top[1])
            candidate_set = []
            for n in range(self.order):
                if n not in set_nodes:
                    continue
                if n not in ptr or self.plug_in_oracle.lookup(n, top[1])[0] < heap[ptr[n]][0]:
                    candidate_set.append(n)
            for c in candidate_set:
                if self.plug_in_oracle.is_uncalculated(c, top[1]):
                    dist = self.oracle(c, top[1])
                    self.plug_in_oracle.update((c, top[1]), dist)
                else:
                    dist = self.plug_in_oracle.lookup(c, top[1])[0]
                if c not in ptr:
                    self._insert(c, dist, top[1], ptr, heap)
                elif dist < heap[ptr[c]][0]:
                    self._update(c, dist, top[1], ptr, heap)
        # print("PRIMS: ", total)
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
            item = heap.pop()
            del ptr[item[1]]
            return item
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