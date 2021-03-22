import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

def draw_graph():
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(15, 11))
    ax1 = fig.add_subplot(1,1,1)
    
    def animate(i):
        RL_types = [3]
        colors = ['red']
        ax1.clear()

        for type_ in RL_types:
            score_file_name = "scores_type" + str(type_) + ".txt"
            graph_data = open(score_file_name, 'r').read()
            lines = graph_data.split('\n')
            xs, ys = [], []
            for line in lines:
                if len(line) > 1:
                    x, y = line.split(',')
                    if (len(xs) - 1 >= 0 and xs[len(xs) - 1] < float(x)) or len(xs) == 0:
                        xs.append(float(x))
                        ys.append(float(y))
        
            ax1.plot(xs[:], ys[:], linewidth=2, alpha=0.3, label="RL agent policy: type" + str(type_), color='red')
            avg_over = 50
            xs_avg = xs[avg_over:]
            ys_avg = []
            for i in range(avg_over, avg_over+len(ys[avg_over:])):
                ys_avg.append(np.mean(ys[i-avg_over:i]))
            ax1.plot(xs_avg, ys_avg, linewidth=2, alpha=1, label="avg score across 50 episodes", color='red')
        
        ax1.set_title("actor_critic agent policy: Scores vs. episodes")
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Total reward per episode')
        ax1.legend()

        plt.savefig('../../plots/Training plots/training_results_actor_critic.png', bbox_inches='tight', )

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

draw_graph()