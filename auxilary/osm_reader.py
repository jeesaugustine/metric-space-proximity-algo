import xml.dom.minidom
from bs4 import BeautifulSoup
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pareto
import statistics
import gmplot


class Node:
    def __init__(self, literal, lat, lon, id):
        self.literal = literal
        self.lat = lat
        self.lon = lon
        self.id = id


def get_nodes_from_xml(file_name):
    doc = xml.dom.minidom.parse(file_name)
    expertise = doc.getElementsByTagName("node")
    nodes = list()
    node_id = 0
    index_list = dict()
    for each in expertise:
        nodes.append(Node(each.getAttribute("id"), each.getAttribute("lat"), each.getAttribute("lon"), node_id))
        index_list[each.getAttribute("id")] = node_id
        node_id += 1
    return nodes, index_list


def get_paths(file_name):
    with open(file_name) as fp:
        soup = BeautifulSoup(fp, 'xml')
    routes = list()
    for each in soup.find_all("way"):
        routes.append([str(e).split('\"')[1] for e in each.contents if str(e)[1:3] == 'nd'])
    return routes


def real_pareto(distance_list, nodes):
    alpha = [1]  # list of values of shape parameters
    samples = np.linspace(start=0, stop=5, num=len(distance_list))
    x_m = 1  # scale
    output = None
    for a in alpha:
        output = np.array([pareto.pdf(x=samples, b=a, loc=0, scale=x_m)])
    plot_graph(output[0], nodes, 'Actual Pareto (auto bin size)', file_name="ParetoDistn.png")


def get_distance_from_edge_info(nodes, routes, index_list, graph_name, small_graph=False):
    graph_real_traffic = open(graph_name, 'w')
    graph_real_traffic.write(str(len(nodes)) + "\n")
    R = 6371e3  # Radius of earth
    distance_dictionary = dict()
    distance_list = list()
    for route in routes:
        for (u, v) in zip(route[:-1], route[1:]):
            if index_list[v] == index_list[u]:
                print("Same id")
            elif index_list[v] > index_list[u]:
                u, v = v, u
            phi1 = float(nodes[index_list[u]].lat) * math.pi / 180  # φ, λ in radians
            phi2 = float(nodes[index_list[v]].lat) * math.pi / 180
            delta_phi = (float(nodes[index_list[v]].lat) - float(nodes[index_list[u]].lat)) * math.pi / 180
            delta_lam = (float(nodes[index_list[v]].lon) - float(nodes[index_list[u]].lon)) * math.pi / 180
            a = (math.sin(delta_phi / 2) * math.sin(delta_phi / 2)) + (math.cos(phi1) * math.cos(phi2)) * (
                    math.sin(delta_lam / 2) * math.sin(delta_lam / 2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            d = R * c / 1000


            if distance_dictionary.get((index_list[u], index_list[v]), -1) == -1:
                if small_graph:
                    if index_list[u] < 100 and index_list[v] < 100:
                        distance_dictionary[(index_list[u], index_list[v])] = d
                        distance_list.append(d)
                        graph_real_traffic.write("{} {} {}\n".format(index_list[u], index_list[v], d))
                else:
                    distance_dictionary[(index_list[u], index_list[v])] = d
                    distance_list.append(d)
                    graph_real_traffic.write("{} {} {}\n".format(index_list[u], index_list[v], d))
    # print("Jees", len(distance_list))
    graph_real_traffic.close()
    return distance_dictionary, distance_list


def plot_on_map(distance_dict, nodes, indexer, file_name):
    lats = list()
    lons = list()

    f = open(file_name, 'w')

    for edge in distance_dict:
        lats.append(float(nodes[edge[0]].lat))
        lats.append(float(nodes[edge[1]].lat))
        lons.append(float(nodes[edge[0]].lon))
        lons.append(float(nodes[edge[1]].lon))
        f.write("{}, {}, {}, {}\n".format(lats[-2], lons[-2], lats[-1], lons[-1], ))
    f.close()
    get_the_map_plotted(lats, lons)


def get_the_map_plotted(lats, lons):
    gmap4 = gmplot.GoogleMapPlotter(statistics.mean(lats), statistics.mean(lons), 5)
    # gmap = gmplot.GoogleMapPlotter(statistics.mean(lats), statistics.mean(lons), 5)
    # gmap4.scatter(lats, lons, '#FF6347', size=40, marker=False)
    # gmap.apikey = "get your api"
    # gmap.scatter(lats, lons, '#FF0000', size=50, marker=False)
    # # Plot method Draw a line in between given coordinates
    # gmap.plot(lats, lons, 'cornflowerblue', edge_width=3.0)
    # Your Google_API_Key
    gmap4.draw('some_new.html')


def plot_graph(distance_list, nodes, title, file_name, if_show=False):
    print("max: {}, min: {}".format(max(distance_list), min(distance_list)))
    bins = np.linspace(math.ceil(min(distance_list)),
                       math.floor(max(distance_list)),
                       20)  # fixed number of bins

    plt.xlim([min(distance_list) - 0.1, max(distance_list) + 0.1])

    print("Number of Nodes: {} and No of Edges: {}".format(len(nodes), len(distance_list)))
    plt.hist(distance_list, bins='auto', alpha=0.25)
    plt.title(title)
    plt.xlabel('variable distance (bin size = auto)')
    plt.ylabel('count')

    if if_show:
        plt.show()
    plt.savefig(file_name)
    plt.close()


def main():
    input_file = "map_ny_new.osm"
    nodes, index_list = get_nodes_from_xml(file_name=input_file)
    routes = get_paths(file_name=input_file)
    output_file_name = "{}_{}.txt".format(input_file.split('.')[0], len(nodes))

    distance_dictionary, distance_list = get_distance_from_edge_info(nodes, routes, index_list,
                                                                     graph_name=output_file_name)
    # distance_dictionary, distance_list = get_distance_from_edge_info(nodes, routes, index_list,
    #                                                                  graph_name='real_traffic_under_100.txt', samll_graph=True)

    plot_name = input_file.split('.')[0] + '.png'
    plot_graph(distance_list, nodes, 'Road Network Data from Open Street View (auto bin size)', file_name=plot_name)

    # real_pareto(distance_list, nodes)
    # plot_on_map(distance_dict=distance_dictionary, nodes=nodes, indexer=index_list, file_name='for_plottng.csv')


if __name__ == "__main__":
    main()
