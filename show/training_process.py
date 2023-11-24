import json
from matplotlib import pyplot as plt
from matplotlib import ticker


def format_func(value, tick_number):
    if value > 0:
        return f'{value / 1000}K'
    else:
        return str(value)


def plot_training(stu_path,tea_path):
    with open(stu_path,'r') as f:
        stu_info = json.load(f)
    with open(tea_path,'r') as f:
        tea_info = json.load(f)
    total_epoch = len(stu_info)
    stu_oks = []
    tea_oks = []
    for i in range(total_epoch):
        stu_oks.append(stu_info[i][2])
        tea_oks.append(tea_info[i][2])

    xlabel = range(0, total_epoch)
    xlabel_transformed = [x * 150 for x in xlabel]
    formatter = ticker.FuncFormatter(format_func)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(formatter)

    ax.plot(xlabel_transformed,stu_oks,label='OKS of student model')
    ax.plot(xlabel_transformed,tea_oks,label='OKS of teacher model')

    ax.set_xlabel("Steps",fontsize=18)
    ax.set_ylabel("OKS performance",fontsize=18)
    ax.tick_params(axis='x',labelsize=14)
    ax.tick_params(axis='y',labelsize=14)
    ax.set_ylim(0,0.7)
    ax.legend(loc='upper left')
    plt.show()
    print('debug')


if __name__ == '__main__':
    # stu_path = '../statistics_file/ours_summary_Val_OKS_Stu_OKS.json'
    # tea_path = '../statistics_file/ours_summary_Val_OKS_Tea_OKS.json'
    stu_path = '../statistics_file/ours_finetune_summary_Val_OKS_Stu_OKS.json'
    tea_path = '../statistics_file/ours_finetune_summary_Val_OKS_Tea_OKS.json'
    plot_training(stu_path,tea_path)