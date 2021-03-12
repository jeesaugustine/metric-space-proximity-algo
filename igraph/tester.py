from helper_igraph import *

if __name__ == "__main__":

    helper_all_generator_methods(128)
    convert_graps_2_distance_dict(graph_size=128)
    # for i in [64, 128, 256, 512]:
    #     real_grah_convert_graps_2_distance_dict(order_val=i, name_of_real="data_20.npy")
    #     real_grah_convert_graps_2_distance_dict(order_val=i, name_of_real="data_sf.npy")
    #     real_grah_convert_graps_2_distance_dict(order_val=i, name_of_real="data_flicker.npy")
    # convert_graps_2_distance_dict(graph_size=8000)

