from graph_maker import NlogNGraphMaker
from sasha_wang import SashaWang

def helper_prims_plugin(num_nodes):
    print(num_nodes)
    graph_maker = NlogNGraphMaker(num_nodes)
    graph = graph_maker.get_nlogn_edges()
    obj_sw = SashaWang()
    obj_sw.store(graph, num_nodes)

if __name__ == "__main__":
    helper_prims_plugin(100)
    helper_prims_plugin(256)
    helper_prims_plugin(512)
    helper_prims_plugin(1000)
    helper_prims_plugin(2000)
    helper_prims_plugin(4000)
