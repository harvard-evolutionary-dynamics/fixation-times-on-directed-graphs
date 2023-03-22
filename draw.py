import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm

from moran import Moran, Type

TIME_STEPS = 200

def animate_nodes(population: Moran, pos=None, *args, **kwargs):
  # define graph layout if None given
  if pos is None:
    pos = nx.planar_layout(population.graph)

  edges = nx.draw_networkx(population.graph, pos, *args, **kwargs, arrows=True, with_labels=False)
  plt.legend()

  def update(ii):
    if ii > 0:
      population._step()

    # draw graph
    nodes = []
    for individual_type, color in zip(Type, 'rb'):
      nodes.append(
        nx.draw_networkx_nodes(population.graph, pos, node_color=color, nodelist=[
            node
            for node, data in population.graph.nodes(data=True)
            if data['type'] == individual_type
          ],
          label=f'{individual_type}',
          *args,
          **kwargs,
        )
      )
    plt.axis('off')

    return nodes

  fig = plt.gcf()
  animation = FuncAnimation(fig, update, interval=5, frames=tqdm(range(TIME_STEPS), initial=1), blit=True)
  return animation


def save_animation(out_file: str) -> None:
  """out_file: gif"""
  animation = animate_nodes(G)
  animation.save(out_file, writer='pillow', savefig_kwargs={'facecolor':'white'}, fps=10)