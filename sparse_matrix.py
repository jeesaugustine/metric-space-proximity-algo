class SparseMatrix:
    def __init__(self, dist_dictionary, order):
        self.distance_hash = {}
        self.neighbours = {}
        self.distances = {}
        self.csr = None
        self.order = order
        self.make_csr_matrix(dist_dictionary)

    def get_row_data(self, row):
        return self.distances[row], self.neighbours[row]

    def get_element(self, x, y):
        return self.distance_hash[(min(x,y), max(x,y))]

    def _put(self, key, value, distance):
        if key in self.neighbours:
            self.neighbours[key].append(value)
            self.distances[key].append(distance)
        else:
            self.neighbours[key] = [value]
            self.distances[key] = [distance]

    def make_csr_matrix(self, dist_dictionary):
        for key,value in dist_dictionary.items():
            x,y = key
            self.distance_hash[(min(x, y), max(x, y))] = value
            self._put(x, y, value)
            self._put(y, x, value)
