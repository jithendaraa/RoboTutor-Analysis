import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

def draw_graph():
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(15, 11))
    ax1 = fig.add_subplot(1,1,1)
    
    def animate(i):
        RL_types = [1,2,3,4,5]
        colors = ['red', 'blue', 'black', 'green', 'purple']
        ax1.clear()

        for type_ in RL_types:
            
            reward_file_name = "test_run_type" + str(type_) + ".txt"
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
        ax1.set_ylabel('Rewards')
        ax1.legend()

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


draw_graph()