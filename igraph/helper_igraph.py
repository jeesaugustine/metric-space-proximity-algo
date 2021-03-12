from standard_graph_generator import StandardGraphGenerator
from igraph_to_distance_graphs import iGraph2DistGraph
import numpy as np
from math import log, ceil

def helper_all_generator_methods(nodes):
    # nodes = 128
    obj = StandardGraphGenerator(nodes)
    prob = ceil(log(nodes, 2))/nodes
    write_status = True

    obj.get_geometric_graph(radius=0.20, torus=True)
    print("Geormetric Connected: ", obj.check_one_connected())
    obj.check_and_write('Geometric_' + str(nodes) + '.pkl', write=obj.check_one_connected())
    # obj.graph_visualize()

    obj.get_barabasi(connectivity=40)
    print("Barabasi Connected: ", obj.check_one_connected())
    obj.check_and_write('Barabasi_' + str(nodes) + '.pkl', write=obj.check_one_connected())
    # obj.graph_visualize()

    obj.get_erods_renyi(prob=prob)
    print("Renyi Erdos Connected: ", obj.check_one_connected())
    obj.check_and_write('Renyi Erdos_' + str(nodes) + '.pkl', write=obj.check_one_connected())
    # obj.graph_visualize()

    obj.get_forrest_fire(fw_prob=prob)
    print("Forrest Fire Connected: ", obj.check_one_connected())
    obj.check_and_write('ForrestFire_' + str(nodes) + '.pkl', write=obj.check_one_connected())
    # obj.graph_visualize()

def convert_graps_2_distance_dict(graph_size):
    file_name = ['Geometric_', 'Barabasi_', 'Renyi Erdos_', 'ForrestFire_']
    mu = 0
    sigma = 0.3
    dim = 4
    a = 2.
    normal = {}
    uniform = {}
    zipf = {}
    ig2dg = iGraph2DistGraph()
    for file in file_name:
        file = file + str(graph_size) + '.pkl'
        print("--" * 2, file, "--" * 2)
        ig2dg.write_status = True
        ig2dg.set_graph(file)
        print("Graph Read Complete")
        file_name = file.split('.')[0].strip()
        ig2dg.add_normal_weights(mu=mu, sigma=sigma, dim=dim, file_name=file_name)
        print("Normal Distance Complete")
        ig2dg.add_uniform_weights(dim=dim, file_name=file_name)
        print("Uniform Distance Complete")
        ig2dg.add_zipf_weights(a=a, dim=dim, file_name=file_name)
        print("zipf Distance Complete\n\n")
        print("--" * 50)


def real_grah_convert_graps_2_distance_dict(order_val, name_of_real):
    file_name = ['Geometric_', 'Barabasi_', 'Renyi Erdos_', 'ForrestFire_']
    ig2dg = iGraph2DistGraph()
    for file in file_name:
        file = file + str(order_val) + '.pkl'
        print("--" * 2, file, "--" * 2)
        ig2dg.write_status = True
        ig2dg.set_graph(file)
        print("Graph Read Complete")
        file_name = name_of_real.split('.')[0].strip() + '_distances_' + file
        ig2dg.sample_and_add_weights(np.load(name_of_real), order_val, file_name)
        print("--" * 50)