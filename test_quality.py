from helper_quality import helper_tester
import sys

if __name__ == "__main__":
    distance_measure = int(sys.argv[1])
    kind = int(sys.argv[2])
    order_val = int(sys.argv[3])
    # ['normal', 'uniform', 'zipf']
    # ['Geometric', 'Renyi Erdos', 'ForrestFire', 'Barabasi']
    helper_tester(order_val, distance_measure, kind)
