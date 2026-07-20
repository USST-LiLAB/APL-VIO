import matplotlib.pyplot as plt
import numpy as np
import os


def getData(folder):
    filenames = sorted(os.listdir(folder))
    sequences = [filename for filename in filenames]
    costs_result = []
    for sequence in sequences:
        with open(folder + sequence, 'r') as file:
            costs_result.append([float(line.strip()) for line in file])
    return costs_result


def main():
    
    fig = plt.figure(figsize=(20, 14))
    fig.subplots_adjust(left=0.04,right=0.98, hspace=0.6)
    plt.rcParams['font.size'] = 20
    # plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    # plt.rcParams['font.sans-serif']=['Times New Roman']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax = fig.add_subplot(3, 1, 1)
    EuRoC_sequences = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05", "V1_01", "V1_02", "V1_03", "V2_01", "V2_02", "V2_03"]

    step = 5
    x1 = np.arange(1, len(EuRoC_sequences) * step + 1, step)
    x2 = np.arange(2, len(EuRoC_sequences) * step + 2, step)
    x3 = np.arange(3, len(EuRoC_sequences) * step + 3, step)
    x4 = np.arange(4, len(EuRoC_sequences) * step + 4, step)
    
    EuRoC_data_folder = "/home/lilabws001/catkin_ws/src/aplvins/src/APL-VINS/analysis/data/stereo/euroc/"
    EuRoC_LSD_LBD_costs = getData(EuRoC_data_folder + "LSD_LBD_costs/")
    EuRoC_ED_LBD_costs = getData(EuRoC_data_folder + "ED_LBD_costs/")
    EuRoC_LSD_ALOFT_costs = getData(EuRoC_data_folder + "LSD_ALOFT_costs/")
    EuRoC_ED_ALOFT_costs = getData(EuRoC_data_folder + "ED_ALOFT_costs/")

    for i in range(0, 11):
        print(np.mean(EuRoC_ED_ALOFT_costs[i]))

    EuRoC_box1 = ax.boxplot(EuRoC_LSD_LBD_costs,
                      positions=x1, 
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#A4DDD3", 
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    EuRoC_box2 = ax.boxplot(EuRoC_ED_LBD_costs, 
                      positions=x2,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#81B21F",
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    EuRoC_box3 = ax.boxplot(EuRoC_LSD_ALOFT_costs, 
                      positions=x3,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#FFDD8E",
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    EuRoC_box4 = ax.boxplot(EuRoC_ED_ALOFT_costs, 
                      positions=x4,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#EB6E60",
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )



    # legend = ax.legend(handles=[EuRoC_box1['boxes'][0], EuRoC_box2['boxes'][0], EuRoC_box3['boxes'][0], EuRoC_box4['boxes'][0]], 
    #           labels=["LSD+LBD", "PLSD+LBD", "LSD+ALOFT", "EDLines+ALOFT"], loc=(0.9, 0.75), ncol=4)
    legend = ax.legend(handles=[EuRoC_box1['boxes'][0], EuRoC_box2['boxes'][0], EuRoC_box3['boxes'][0], EuRoC_box4['boxes'][0]], 
              labels=["LSD+LBD", "EDLines+LBD", "LSD+ALOFT", "EDLines+ALOFT"], loc="upper right", ncol=4, fontsize="16")
    ax.set_xlim(0, len(EuRoC_sequences) * step)
    ax.set_ylim(-10, 110)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(step))
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(True, axis='x', which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, axis='y', which='major', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks((x1 + x4) / 2, [seq[0:5] for seq in EuRoC_sequences])
    ax.tick_params(direction='in')
    ax.set_xlabel("EuRoC datasets", fontsize=28)
    ax.set_ylabel("Runtimes[ms]")


    ax2 = fig.add_subplot(3, 1, 2)

    TUMVI_data_folder = "/home/lilabws001/catkin_ws/src/aplvins/src/APL-VINS/analysis/data/stereo/tumvi/"
    TUMVI_LSD_LBD_costs = getData(TUMVI_data_folder + "LSD_LBD_costs/")
    TUMVI_ED_LBD_costs = getData(TUMVI_data_folder + "ED_LBD_costs/")
    TUMVI_LSD_ALOFT_costs = getData(TUMVI_data_folder + "LSD_ALOFT_costs/")
    TUMVI_ED_ALOFT_costs = getData(TUMVI_data_folder + "ED_ALOFT_costs/")

    TUMVI_box1 = ax2.boxplot(TUMVI_LSD_LBD_costs,
                      positions=x1, 
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#A4DDD3", 
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    TUMVI_box2 = ax2.boxplot(TUMVI_ED_LBD_costs,
                      positions=x2, 
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#81B21F", 
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    TUMVI_box3 = ax2.boxplot(TUMVI_LSD_ALOFT_costs, 
                      positions=x3,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#FFDD8E",
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    TUMVI_box4 = ax2.boxplot(TUMVI_ED_ALOFT_costs, 
                      positions=x4,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#EB6E60",
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )


    TUMVI_sequences = ["Cor_1", "Cor_2", "Cor_3", "Cor_4", "Cor_5", "Mag_1", "Mag_2", "Mag_3", "Mag_4", "Mag_5", "Mag_6"]
    ax2.set_xlim(0, len(TUMVI_sequences) * step)
    ax2.set_ylim(-10, 70)
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(step))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax2.grid(True, axis='x', which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.grid(True, axis='y', which='major', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xticks((x1 + x4) / 2, [seq[0:5] for seq in TUMVI_sequences])
    ax2.tick_params(direction='in')
    ax2.set_xlabel("TUM VI datasets", fontsize=28)
    ax2.set_ylabel("Runtimes[ms]")



    ax3 = fig.add_subplot(3, 1, 3)

    KAIST_data_folder = "/home/lilabws001/catkin_ws/src/aplvins/src/APL-VINS/analysis/data/stereo/kaist/"
    KAIST_LSD_LBD_costs = getData(KAIST_data_folder + "LSD_LBD_costs/")
    KAIST_ED_LBD_costs = getData(KAIST_data_folder + "ED_LBD_costs/")
    KAIST_LSD_ALOFT_costs = getData(KAIST_data_folder + "LSD_ALOFT_costs/")
    KAIST_ED_ALOFT_costs = getData(KAIST_data_folder + "ED_ALOFT_costs/")

    KAIST_box1 = ax3.boxplot(KAIST_LSD_LBD_costs,
                      positions=x1, 
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#A4DDD3", 
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    KAIST_box2 = ax3.boxplot(KAIST_ED_LBD_costs,
                      positions=x2, 
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#81B21F", 
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    KAIST_box3 = ax3.boxplot(KAIST_LSD_ALOFT_costs, 
                      positions=x3,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#FFDD8E",
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    KAIST_box4 = ax3.boxplot(KAIST_ED_ALOFT_costs, 
                      positions=x4,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#EB6E60",
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )


    KAIST_sequences = ["Cir_f", "Cir_h", "Cir_n", "Inf_f", "Inf_h", "Inf_n", "Rot_f", "Rot_n", "Squ_f", "Squ_h", "Squ_n"]
    ax3.set_xlim(0, len(KAIST_sequences) * step)
    ax3.set_ylim(-10, 70)
    ax3.xaxis.set_minor_locator(plt.MultipleLocator(step))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax3.grid(True, axis='x', which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.grid(True, axis='y', which='major', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xticks((x1 + x4) / 2, [seq[0:5] for seq in KAIST_sequences])
    ax3.tick_params(direction='in')
    ax3.set_xlabel("KAIST datasets", fontsize=28)
    ax3.set_ylabel("Runtimes[ms]")



    plt.savefig('/home/lilabws001/catkin_ws/src/aplvins/src/APL-VINS/output/boxPlot_stereo.svg', format='svg')
    plt.show()


if __name__ == "__main__":
    main()
