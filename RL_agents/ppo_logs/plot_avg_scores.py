import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

def draw_graph():
    style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    def animate(i):
        graph_data = open('avg_scores.txt', 'r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        for line in lines:
            if len(line) > 1:
                x, y = line.split(',')
                xs.append(float(x))
                ys.append(float(y))
        ax1.clear()
        ax1.plot(xs, ys, color="purple")
        if len(xs) >= 1 and len(ys) >= 1:
            ax1.plot([xs[0], xs[len(xs) - 1]], [ys[0], ys[0]], color="black")
            ax1.plot([xs[0], xs[len(xs) - 1]], [ys[0], ys[len(ys) - 1]], color="yellow")

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

draw_graph()