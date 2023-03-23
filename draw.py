import networkx as nx
import time
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt; plt.close('all')
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
import itertools
from stats import is_absorbed
import utils
from halo import Halo

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
    for is_active in (False, True):
      edgelist=[
        (u, v)
        for u, v, active in population.graph.edges.data(Moran.IS_ACTIVE_EDGE_ATTRIBUTE_NAME, default=False)
        if active == is_active
      ]
      nx.draw_networkx_edges(
        population.graph,
        pos,
        *args,
        **kwargs,
        edge_color=('black', 'orange')[is_active],
        edgelist=edgelist,
        arrows=True,
      )
    
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
            (color, 'green', 'purple')[
                 (data.get(Moran.IS_BIRTHING_ATTRIBUTE_NAME) and 1)
              or (data.get(Moran.IS_DYING_ATTRIBUTE_NAME) and 2)
              or 0
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
  with Halo('animating nodes') as spinner:
    animation = animate_nodes(population)
    spinner.succeed()

  start = time.time()
  with Halo('saving animation') as spinner:
    animation.save(out_file, writer='pillow', savefig_kwargs={'facecolor':'white'}, fps=10, dpi=500)
    end = time.time()
    elapsed_seconds = end - start
    spinner.succeed(f"Done in {elapsed_seconds:.2f}s")