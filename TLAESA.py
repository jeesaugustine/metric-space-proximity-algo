import random
import time
from LAESA import NodeLandMarkRandom

def select_the_new_landmark(sum_node_to_landmark):
    max_key = max(sum_node_to_landmark, key=sum_node_to_landmark.get)

    return max_key


class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.rep = None
        self.table = dict()
        self.tree_rep = None
        

class PTTree:
    def __init__(self, oracle, laesa, order_val, start_node=None):
        self.oracle = oracle
        self.laesa = laesa
        self.tree = Tree()
        self.order_val = order_val
        self.loc = [None]*order_val
        self.tree_rep = start_node

    def built_tree(self):
        S = list(range(self.order_val))
        if self.tree_rep == None: 
            self.tree_rep = random.choice(S)
        t = self.tree.rep = self.tree_rep
        self.loc[t] = self.tree
        self.tree.table[t] = 0
        if t in self.laesa.k_landmarks:
            max_dist = -1
            max_node = None
            for i in S:
                if i == t: continue
                dist = self.laesa.lookup(t, i)[0]
                self.tree.table[i] = dist
                if max_dist < dist:
                    max_dist = dist
                    max_node = i
        else:
            max_dist = -1
            max_node = None
            for i in S:
                if i == t: continue
                if i in self.laesa.k_landmarks:
                    dist = self.laesa.lookup(t, i)[0]
                else:
                    dist = self.oracle(t, i)
                    
                self.tree.table[i] = dist
                if max_dist < dist:
                    max_dist = dist
                    max_node = i
        self.second_build_tree(max_node, self.tree, S)

    def handle_right(self, parent, S):
        if len(S) <= 1:
            if len(S) == 1:
                self.loc[S[0]] = parent
            return
        new_tree_right = Tree()
        parent.right = new_tree_right
        new_tree_right.rep = parent.rep
        new_tree_right.table = parent.table
        new_tree_right.parent = parent
        m_t, m_dist = -1, -1
        for i in S:
            if i == parent.rep: continue
            if parent.table[i] > m_dist:
                m_dist = new_tree_right.table[i]
                m_t = i
        self.second_build_tree(m_t, new_tree_right, S)

    def second_build_tree(self, max_node, parent, S):
        new_tree_left = Tree()
        left, right = [], []
        m_t, m_dist = -1, -1
        for i in S:
            if i == max_node or i == parent.rep: continue
            if i in self.laesa.k_landmarks or max_node in self.laesa.k_landmarks:
                dist = self.laesa.lookup(max_node, i)[0]
            else:
                dist = self.oracle(max_node, i)
            new_tree_left.table[i] = dist
            dist1 = parent.table[i]
            if dist < dist1:
                left.append(i)
                if m_dist < dist:
                    m_dist = dist
                    m_t = i
            else:
                right.append(i)
        parent.left = new_tree_left
        new_tree_left.rep = max_node
        new_tree_left.table[max_node] = 0
        new_tree_left.parent = parent
        self.loc[max_node] = new_tree_left
        if len(left) > 1:
            self.second_build_tree(m_t, parent.left, left)
        elif len(left) == 1:
            self.loc[left[0]] = new_tree_left
        self.handle_right(parent, right)

    def get_path_length(self, node):
        length = 0
        while node:
            length += 1
            node = node.parent
        return length

    def lookup(self, x, y):
        len_x = self.get_path_length(self.loc[x])
        len_y = self.get_path_length(self.loc[y])
        min_len = min(len_x, len_y)
        loc_x, loc_y = self.loc[x], self.loc[y]
        while len_x > min_len:
            loc_x = loc_x.parent
            len_x -= 1
        while len_y > min_len:
            loc_y = loc_y.parent
            len_y -= 1
        first = True
        lb_val, ub_val = 0, 1
        while loc_x and loc_y:
            if loc_x == loc_y:
                if loc_x.rep == x: return [loc_x.table[y], loc_x.table[y]]
                if loc_x.rep == y: return [loc_x.table[x], loc_x.table[x]]
                lb_val = max(lb_val, abs(loc_x.table[x] - loc_x.table[y]))
                ub_val = min(ub_val, loc_x.table[x] + loc_x.table[y])
                if first:
                    first = False
                    if loc_x.left:
                        lb_val = max(lb_val, abs(loc_x.left.table[x] - loc_x.left.table[y]))
                        ub_val = min(ub_val, loc_x.left.table[x] + loc_x.left.table[y])
            loc_x = loc_x.parent
            loc_y = loc_y.parent
        return [lb_val, ub_val]

    def is_uncalculated(self, x, y):
        loc_x = self.loc[x]
        while loc_x:
            if loc_x.rep == x:
                return y not in loc_x.table
            if loc_x.rep == y:
                return x not in loc_x.table
            loc_x = loc_x.parent
        return True


class TLAESA:
    def __init__(self, g, k, oracle, order_val, start_node=None):
        self.laesa = NodeLandMarkRandom(g, k, oracle, order_val)
        self.pttree = PTTree(oracle, self.laesa, order_val, start_node)
        self.pttree.built_tree()

    def lookup(self, x, y):
        laesa_vals = self.laesa.lookup(x, y)
        pttree_vals = self.pttree.lookup(x, y)
        ret = [max(laesa_vals[0], pttree_vals[0]), min(laesa_vals[1], pttree_vals[1])]
        return ret

    def update(self, edge, val):
        self.laesa.update(edge, val)

    def is_uncalculated(self, x, y):
        x, y = min(x, y), max(x, y)
        return self.laesa.is_uncalculated(x, y) and self.pttree.is_uncalculated(x, y)

