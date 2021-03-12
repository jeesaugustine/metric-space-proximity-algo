def _get_dbl_level_dict(g_dict):
    dbl_dict = {}
    for k, val in g_dict.items():
        i, j = k
        if i not in dbl_dict:
            dbl_dict[i] = {}
        if j not in dbl_dict:
            dbl_dict[j] = {}
        dbl_dict[i][j] = val
        dbl_dict[j][i] = val
    return dbl_dict