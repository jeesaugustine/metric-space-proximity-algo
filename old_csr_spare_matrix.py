from scipy.sparse import *

class SparseMatrix:
    def __init__(self, dist_dictionary, order):
        self.distance = dist_dictionary
        self.csr = None
        self.order = order
        self.make_csr_matrix()

    def get_row_data(self, row):
        indices = range(self.csr.indptr[row], self.csr.indptr[row + 1])
        distances = self.csr.data[indices]
        neighbours = self.csr.indices[indices]
        return distances, neighbours

    def make_csr_matrix(self):
        rows = []
        cols = []
        data = []
        for each in self.distance:
            rows.append(each[0])
            rows.append(each[1])
            cols.append(each[1])
            cols.append(each[0])
            data.append(self.distance[each])
            data.append(self.distance[each])
        self.csr = csr_matrix((data, (rows, cols)), shape=(self.order, self.order))