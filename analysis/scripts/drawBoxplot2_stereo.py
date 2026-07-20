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
    
    fig = plt.figure(figsize=(20, 6))
    fig.subplots_adjust(left=0.04,right=0.98, hspace=0.4)
    plt.rcParams['font.size'] = 20
    # plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    # plt.rcParams['font.sans-serif']=['Times New Roman']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    ax = fig.add_subplot(2, 1, 1)
    EuRoC_sequences = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05", "V1_01", "V1_02", "V1_03", "V2_01", "V2_02", "V2_03"]

    step = 3
    x1 = np.arange(1, len(EuRoC_sequences) * step + 1, step)
    x2 = np.arange(2, len(EuRoC_sequences) * step + 2, step)
    
    EuRoC_data_folder = "/home/lilabws001/catkin_ws/src/aplvins/src/APL-VINS/analysis/data/stereo/euroc/"
    EuRoC_LSD_LBD_costs = getData(EuRoC_data_folder + "LSD_LBD_costs/")
    EuRoC_PLSD_LBD_costs = getData(EuRoC_data_folder + "PLSD_LBD_costs/")


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

    EuRoC_box2 = ax.boxplot(EuRoC_PLSD_LBD_costs,
                      positions=x2, 
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#81B21F", 
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    legend = ax.legend(handles=[EuRoC_box1['boxes'][0], EuRoC_box2['boxes'][0]], 
              labels=["origin LSD", "Ours"], loc="upper right", ncol=4, fontsize="16")
    ax.set_xlim(0, len(EuRoC_sequences) * step)
    ax.set_ylim(0, 90)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(step))
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(True, axis='x', which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, axis='y', which='major', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks((x1 + x2) / 2, [seq[0:5] for seq in EuRoC_sequences])
    ax.tick_params(direction='in')
    ax.set_xlabel("EuRoC datasets")
    ax.set_ylabel("Runtimes[ms]")


    ax2 = fig.add_subplot(2, 1, 2)

    TUMVI_data_folder = "/home/lilabws001/catkin_ws/src/aplvins/src/APL-VINS/analysis/data/stereo/tumvi/"
    TUMVI_LSD_LBD_costs = getData(TUMVI_data_folder + "LSD_LBD_costs/")
    TUMVI_PLSD_LBD_costs = getData(TUMVI_data_folder + "PLSD_LBD_costs/")


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

    TUMVI_box2 = ax2.boxplot(TUMVI_PLSD_LBD_costs,
                      positions=x2, 
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor="#81B21F", 
                                    edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black")
                      )

    TUMVI_sequences = ["Cor_1", "Cor_2", "Cor_3", "Cor_4", "Cor_5", "Mag_1", "Mag_2", "Mag_3", "Mag_4", "Mag_5", "Mag_6"]
    ax2.set_xlim(0, len(TUMVI_sequences) * step)
    ax2.set_ylim(0, 70)
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(step))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax2.grid(True, axis='x', which='minor', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.grid(True, axis='y', which='major', color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xticks((x1 + x2) / 2, [seq[0:5] for seq in TUMVI_sequences])
    ax2.tick_params(direction='in')
    ax2.set_xlabel("TUM VI datasets")
    ax2.set_ylabel("Runtimes[ms]")


    plt.savefig('/home/lilabws001/catkin_ws/src/aplvins/src/APL-VINS/output/boxPlot.pdf', dpi=300, format='pdf')
    plt.show()


if __name__ == "__main__":
    main()
