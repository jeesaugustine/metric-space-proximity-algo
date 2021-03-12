def sw_single_shot(double** lb_matrix,double** ub_matrix,int n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                lb_matrix[i][j] = lb_matrix[j][i] = max(lb_matrix[i][j], lb_matrix[i][k] - ub_matrix[k][j],
                                          lb_matrix[j][k] - ub_matrix[k][i])
                ub_matrix[i][j] = ub_matrix[j][i] = min(ub_matrix[i][j], ub_matrix[i][k] + ub_matrix[k][j])
"""

def _sw_update(np.ndarray matrix,np.ndarray lb_matrix,np.ndarray ub_matrix, np.ndarray uncalculated, int x, int y, double d):
    matrix[x][y] = matrix[y][x] = lb_matrix[x][y] = lb_matrix[y][x] = ub_matrix[x][y] = ub_matrix[y][x] = d
    # Perform uncalculated outside
    # uncalculated[min(x, y)].remove(max(x, y))
    # if len(uncalculated[min(x, y)]) == 0:
    #     del uncalculated[min(x, y)]
    for i in range(uncalculated.shape[0]):
        lb_matrix[i[0]][i[1]] = lb_matrix[i[1]][i[0]] = max(
            lb_matrix[i[0]][i[1]],
            d - ub_matrix[i[0]][x] - ub_matrix[i[1]][y],
            d - ub_matrix[i[0]][y] - ub_matrix[i[1]][x],
            lb_matrix[i[0]][x] - d - ub_matrix[i[1]][y],
            lb_matrix[i[0]][y] - d - ub_matrix[i[1]][x],
            lb_matrix[i[1]][x] - d - ub_matrix[i[0]][y],
            lb_matrix[i[1]][y] - d - ub_matrix[i[0]][x]
        )
        ub_matrix[i[0]][i[1]] = ub_matrix[i[1]][i[0]] = min(ub_matrix[i[1]][i[0]],
                ub_matrix[i[0]][x] + d + ub_matrix[y][i[1]],
                ub_matrix[i[0]][y] + d + ub_matrix[x][i[1]])
"""
