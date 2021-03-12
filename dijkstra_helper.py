from dijkstra import Dijkstra
from graph_maker import NlogNGraphMaker
from helper import _get_dbl_level_dict

order_val = 1000
graph_maker = NlogNGraphMaker(order_val)
g = graph_maker.get_nlogn_edges()
dbl_dict = _get_dbl_level_dict(g)

d = Dijkstra(dbl_dict, order_val)
print(d.shortest_path(list(range(order_val)), 0))
