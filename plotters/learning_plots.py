import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

# Rewards vs epochs plots: Same type, across policies
root_path = '../logs/'
algos = ['dqn', 'actor_critic', 'ppo']
colors = ['red', 'blue', 'green']
types = ['3']

style.use('fivethirtyeight')
fig = plt.figure(figsize=(15, 11))
ax1 = fig.add_subplot(1,1,1)


def plot_learning_across_rl_algos():

    def animate(k):
        ax1.clear()
        for type_ in types:
            learning_files = []

            for algo in algos:
                if algo == 'ppo':   learning_files.append(root_path + algo + '_logs/test_run_type' + type_ + '.txt')
                else:               learning_files.append(root_path + algo + '_logs/scores_type' + type_ + '.txt')

            for i in range(len(learning_files)):
                learning_file = learning_files[i]
                lines = open(learning_file, 'r').read().split('\n')
                xs, ys = [], []
                for line in lines:
                    if len(line) > 1:
                        x, y = line.split(',')
                        if (len(xs) - 1 >= 0 and xs[len(xs) - 1] < float(x)) or len(xs) == 0:
                            xs.append(float(x))
                            ys.append(float(y))

                xs = xs[:]
                ys = ys[:]
                avg_over = 30
                ys_avg = []
                xs_avg = xs[avg_over:]
                for j in range(avg_over, avg_over+len(ys[avg_over:])):
                    ys_avg.append(np.mean(ys[j-avg_over:j]))

                ax1.plot(xs[:], ys[:], linewidth=2, alpha=0.3, label= "Type " + str(type_) + ":" + algos[i], color=colors[i])
                ax1.plot(xs_avg, ys_avg, linewidth=2, alpha=1, label= "Type " + str(type_) + ":" + algos[i] + " Running average across " + str(avg_over) + " epochs", color=colors[i])

            ax1.set_title("(type " + type_ + ") Rewards vs. epochs across deep RL policies during policy learning")
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Rewards')
            ax1.legend()

            plt.savefig('../plots/Training plots/learning_plots_type'+type_+'.png', bbox_inches='tight', )


    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


plot_learning_across_rl_algos()
