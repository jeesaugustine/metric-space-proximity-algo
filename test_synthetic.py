from sasha_wang import SashaWang
from helper import *
from parametrized_path_search import ParamTriSearch
from helper_other_algos import *
from helper_plug_in_synthetic_graphs import *
import sys

if __name__ == "__main__":

    size_of_graph = int(sys.argv[1])
    type_of_graph = int(sys.argv[2])
    helper_sasha_wang_saver(size_of_graph, type_of_graph)

    # First Shot
    # helper_sasha_wang_saver(0, 0)
    # helper_sasha_wang_saver(0, 1)
    # helper_sasha_wang_saver(0, 2)
    # helper_sasha_wang_saver(0, 3)
    # helper_sasha_wang_saver(1, 0)
    # helper_sasha_wang_saver(1, 1)
    # helper_sasha_wang_saver(1, 2)

    # Second shot

    # helper_sasha_wang_saver(1, 3)
    # helper_sasha_wang_saver(2, 0)
    # helper_sasha_wang_saver(2, 1)
    # helper_sasha_wang_saver(2, 2)
    # helper_sasha_wang_saver(2, 3)