class Node:
    def __init__(self, index, distance, parent, children=[]):
        self.index = index
        self.distance = distance
        self.max = -1
        self.path_length = -1
        self.parent = parent
        self.children = children

    def __str__(self):
        return str((self.index, self.distance, self.max, self.path_length, self.children))
