class Dijkstra:
    def __init__(self, dist_dict, order):
        self.dist_dict = dist_dict
        self.order = order

    def shortest_path(self, nodes, source):
        sp = dict(zip(nodes, [[1] for i in range(self.order)]))
        set_nodes = set(nodes)
        # (distance, cur_node, parent)
        heap = [[0, source, source]]
        ptr = {}
        while heap:
            top = self._heap_pop(heap, ptr)
            set_nodes.remove(top[1])
            cur_length = top[0]
            sp[top[1]] = [cur_length, top[2]]
            neighbours = set(self.dist_dict[top[1]].keys()).intersection(set_nodes)
            for n in neighbours:
                if n in ptr and heap[ptr[n]][0] > self.dist_dict[top[1]][n] + cur_length:
                    self._update(n, self.dist_dict[top[1]][n] + cur_length, top[1], ptr, heap)
                elif n not in ptr:
                    self._insert(n, self.dist_dict[top[1]][n] + cur_length, top[1], ptr, heap)
        return sp

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