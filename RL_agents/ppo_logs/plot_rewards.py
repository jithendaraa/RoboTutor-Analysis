import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

def draw_graph():
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(15, 11))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    def animate(i):
        RL_types = [1,2,3,4,5]
        colors = ['red', 'blue', 'black', 'green', 'purple']
        ax1.clear()

        for type_ in RL_types:
            reward_file_name = "rewards_type" + str(type_) + ".txt"
            graph_data = open(reward_file_name, 'r').read()
            lines = graph_data.split('\n')
            xs, ys = [], []
            for line in lines:
                if len(line) > 1:
                    x, y = line.split(',')
                    if (len(xs) - 1 >= 0 and xs[len(xs) - 1] < float(x)) or len(xs) == 0:
                        xs.append(float(x))
                        ys.append(float(y))
        
            ax1.plot(xs[:], ys[:], linewidth=2, alpha=0.5, label="RL agent policy: type" + str(type_), color=colors[type_ - 1])
        
        ax1.set_title("RL agent policy")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Avg P(Know) across all skills')
        ax1.legend()

        threshold_data = open('current_rt_thresholds.txt', 'r').read()
        lenient_threshold_data = open('current_rt_lenient_thresholds.txt', 'r').read()
        threshold_lines = threshold_data.split('\n')
        lenient_threshold_lines = lenient_threshold_data.split('\n')
        threshold_xs, lenient_threshold_xs = [], []
        threshold_ys, lenient_threshold_ys = [], []
        for line in threshold_lines:
            if len(line) > 1:
                x, y = line.split(',')
                if (len(threshold_xs) - 1 >= 0 and threshold_xs[len(threshold_xs) - 1] < float(x)) or len(threshold_xs) == 0:
                    threshold_xs.append(float(x))
                    threshold_ys.append(float(y))
        for line in lenient_threshold_lines:
            if len(line) > 1:
                x, y = line.split(',')
                if (len(lenient_threshold_xs) - 1 >= 0 and lenient_threshold_xs[len(lenient_threshold_xs) - 1] < float(x)) or len(lenient_threshold_xs) == 0:
                    lenient_threshold_xs.append(float(x))
                    lenient_threshold_ys.append(float(y))
        ax2.clear()
        ax2.plot(threshold_xs, threshold_ys, color='red', label='Current RT Thresholds (0.5, 0.83, 0.9)', linewidth=2)
        ax2.plot(lenient_threshold_xs, lenient_threshold_ys, color='blue', label='Current RT Lenient Thresholds (0.4, 0.55, 0.7)', linewidth=2)
        ax2.set_title("Current RoboTutor threshold policies")
        ax2.set_xlabel('Activity attempts')
        ax2.set_ylabel('Avg P(Know) across all skills')
        ax2.legend()

        plt.savefig('../../plots/Training plots/training_results.png', bbox_inches='tight', )

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


draw_graph()