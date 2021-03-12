def main():
    input_filename = "this_is_the_new_map_ny_445_466.txt"
    output_filename = "new_parsed_out_fix_map_new_445_466.txt"

    with open(input_filename, 'r') as ouput:
        results = ouput.readlines()
    g = (i for i, e in enumerate(results) if e[:15] == "Duration ELM LB")
    # write_content = single_files(g, results)
    write_content = pairs(g, results)
    with open(output_filename, 'w') as f:
        for each in write_content:
            f.write(each)


def pairs(g, results):
    write_content = list()
    g_list = list(g)
    i = 0
    while i < len(g_list):
        Duration_ELM_LB = results[g_list[i]].split(':')[1].strip()
        Total_ELM_LB = results[g_list[i] + 1].split(':')[1].strip()
        Duration_SW_LB = results[g_list[i] + 2].split(':')[1].strip()
        Total_Original_Graph_weight = results[g_list[i] + 3].split(':')[1].strip()
        Total_SW_LB = results[g_list[i] + 4].split(':')[1].strip()
        relative = results[g_list[i] + 5].split(':')[1].strip()
        rmse = results[g_list[i] + 6].split(':')[1].strip()

        i += 1
        try:
            Duration_ELM_LB_1 = results[g_list[i]].split(':')[1].strip()
            Total_ELM_LB_1 = results[g_list[i] + 1].split(':')[1].strip()
            Duration_SW_LB_1 = results[g_list[i] + 2].split(':')[1].strip()
            Total_Original_Graph_weight_1 = results[g_list[i] + 3].split(':')[1].strip()
            Total_SW_LB_1 = results[g_list[i] + 4].split(':')[1].strip()
            relative_1 = results[g_list[i] + 5].split(':')[1].strip()
            rmse_1 = results[g_list[i] + 6].split(':')[1].strip()

        except:
            Duration_ELM_LB_1 = ""
            Total_ELM_LB_1 = ""
            Duration_SW_LB_1 = ""
            Total_Original_Graph_weight_1 = ""
            Total_SW_LB_1 = ""
            relative_1 = ""
            rmse_1 = ""
        i += 1

        write_content.append(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(Duration_ELM_LB, Duration_ELM_LB_1, Total_ELM_LB,
                                                                      Total_ELM_LB_1, Duration_SW_LB, Duration_SW_LB_1,
                                                                      Total_Original_Graph_weight, relative, relative_1,
                                                                      rmse, rmse_1, Total_SW_LB_1))
    return write_content


def single_files(g, results):
    write_content = list()
    for indx in g:
        Duration_ELM_LB = results[indx].split(':')[1].strip()
        Total_ELM_LB = results[indx + 1].split(':')[1].strip()
        Duration_SW_LB = results[indx + 2].split(':')[1].strip()
        Total_Original_Graph_weight = results[indx + 3].split(':')[1].strip()
        Total_SW_LB = results[indx + 4].split(':')[1].strip()
        relative = results[indx + 5].split(':')[1].strip()
        rmse = results[indx + 6].split(':')[1].strip()

        write_content.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(Duration_ELM_LB, Total_ELM_LB, Duration_SW_LB,
                                                                   Total_Original_Graph_weight, Total_SW_LB, relative,
                                                                   rmse))
    return write_content


if __name__ == "__main__":
    main()
