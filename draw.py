import networkx as nx
import matplotlib.pyplot as plt; plt.close('all')
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
import itertools
from stats import is_absorbed
import utils

from moran import Moran, Type

MAX_NUM_TIME_STEPS = 99999999999999

def animate_nodes(population: Moran, pos=None, *args, **kwargs):
  # define graph layout if None given
  if pos is None:
    pos = nx.kamada_kawai_layout(population.graph)

  edges = nx.draw_networkx(population.graph, pos, *args, **kwargs, arrows=True, with_labels=False)
  plt.legend()

  def update(ii):
    # if ii > 0:
    #   population._step()

    # draw graph
    nodes = []
    for individual_type, color in zip(Type, 'br'):
      nodes_with_data = [
        (node, data)
        for node, data in population.graph.nodes(data=True)
        if data['type'] == individual_type
      ]
      nodes.append(
        nx.draw_networkx_nodes(
          population.graph,
          pos,
          node_color=color,
          edgecolors=[
            (color, 'green')[
              data.get(Moran.IS_BIRTHING_ATTRIBUTE_NAME) or data.get(Moran.IS_DYING_ATTRIBUTE_NAME) or False
            ]
            for _, data in nodes_with_data
          ],
          nodelist=[
            node
            for node, _ in nodes_with_data
          ],
          label=f'{individual_type}',
          *args,
          **kwargs,
        )
      )
    plt.axis('off')

    return nodes

  fig = plt.gcf()
  # for idx in enumerate(population.evolve(include_initial=True)):
  #   if is_absorbed(population.graph):
  #     break
  #   print('ho', idx)
  # input('done')
  animation = FuncAnimation(fig, update, interval=2, frames=(
      utils.takewhile_inclusive(
        lambda action_state: not is_absorbed(action_state.state.graph),
        population.evolve(include_initial=True)
      )
      # initial=1,
    ),
    blit=False,
  )
  return animation


def save_animation(population: Moran, out_file: str) -> None:
  """out_file: gif"""
  animation = animate_nodes(population)
  animation.save(out_file, writer='pillow', savefig_kwargs={'facecolor':'white'}, fps=10)