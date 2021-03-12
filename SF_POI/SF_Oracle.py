import math
import pickle


def get_cured_lat_long(name, rest):
    with open(name, "r") as f:
        pois = f.readlines()
    pois = pois[:rest]
    cured_pois = []
    for poi in pois:
        if len(poi.strip()) > 0:
            lat_long = poi.strip().split(" ")
            lat = float(lat_long[0].strip())
            lon = float(lat_long[1].strip())
            if lat < 0:
                lat = 180 - lat
            if lon < 0:
                lon = 180 - lon
            cured_pois.append((lat, lon))
    # pickle.dump(cured_pois, open(name.split(".txt")[0] + ".pkl", 'wb'))
    return cured_pois


def get_curated_lat_long_from_pickle(name):
    return pickle.load(open(name, 'rb'))


def get_dist(p1, p2):
    R = 6371e3
    (lat1, long1) = p1
    (lat2, long2) = p2
    phi1 = lat1 * math.pi / 180
    phi2 = lat2 * math.pi / 180
    delphi = (lat2 - lat1) * math.pi / 180
    dellam = (long2 - long1) * math.pi / 180
    a = math.sin(delphi / 2) * math.sin(delphi / 2) + math.cos(phi1) * \
        math.cos(phi2) * math.sin(dellam / 2) * math.sin(dellam / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def get_SF_ORACLE(name, limit_rows, maximum, debug=False):
    # cured_pois = get_cured_lat_long("CA.txt", rest=10000)
    cured_pois = get_cured_lat_long(name, limit_rows)
    def oracle(u, v):
        if debug:
            print("Point 1: {},\tPoint 2: {},\tDist: {}"
                  .format(cured_pois[u], cured_pois[v], get_dist(cured_pois[u], cured_pois[v]) / (maximum * 1000)))
        # return get_dist(cured_pois[u], cured_pois[v]) / (1201*1000)
        return get_dist(cured_pois[u], cured_pois[v]) / (maximum * 1000)
    return oracle


def get_max2normalise(name, limit_rows, print_interval, mult):
    len_cured_pois = len(get_cured_lat_long(name, limit_rows))
    SF_Oracle = get_SF_ORACLE(name, limit_rows, maximum=1, debug=False)
    maximum = 0
    for i in range(len_cured_pois):
        for j in range(i + 1, len_cured_pois):
            if maximum < SF_Oracle(i, j):
                maximum = SF_Oracle(i, j)
        if i % print_interval == 0:
            print("So far we covered i: {}".format(i))
        if i == mult:
            print("maximum at i: {} is :{}".format(i, maximum))
            mult *= 2
    print(maximum)
    return maximum
