import jax.numpy as jnp
from jraph import GraphsTuple
import jraph
import haiku as hk
import jax
from typing import Tuple, List, Dict, Any
from utils.preprocessing import GraphDataPoint
from tqdm.notebook import tqdm
import functools
import optax
from gnome_model.gnn import GraphNetwork as GnomeGraphNetwork
from gnome_model.crystal import mlp
from functools import partial

def count_params(params:hk.Params):
  w_params = 0
  b_params = 0
  total_params = 0
  for layer in params.keys():
      w_size = params[layer]['w'].shape
      w_param_cnt = w_size[0] * w_size[1]
      w_params += w_param_cnt
      b_param_count = params[layer]['b'].shape[0]
      b_params += b_param_count
      total_params += w_param_cnt + b_param_count

  return total_params, w_params, b_params

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y

def pad_graph_to_nearest_power_of_two(graphs_tuple: GraphsTuple) -> GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two.
  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
  Returns:
    A graphs_tuple batched to the nearest power of two.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Edge update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.swish,
       hk.Linear(128)])
  return net(feats)

@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Node update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.swish,
       hk.Linear(128)])
  return net(feats)

@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  # BandGap Prediction is a regression, so output a single value.
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.swish,])
  return net(feats)

@jraph.concatenated_args
def readout_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  # BandGap Prediction is a regression, so output a single value.
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.swish,
       hk.Linear(1)])
  return net(feats)

def net_fn(graph: jraph.GraphsTuple, steps:int) -> jraph.GraphsTuple:
  # Add a global paramater for graph classification.
  graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
  embedder = jraph.GraphMapFeatures(
      hk.Linear(128), hk.Linear(128), hk.Linear(128))
  net = jraph.GraphNetGAT(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn)
  
  readout = jraph.GraphNetGAT(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=readout_global_fn)
  
  output = embedder(graph)
  for _ in range(steps):
    output = net(output)

  return readout(output)
  
def compute_loss(params: hk.Params, graph: jraph.GraphsTuple, label: jnp.ndarray,
                 net: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes loss and accuracy."""
  pred_graph = net.apply(params, graph)
  # preds = jax.nn.log_softmax(pred_graph.globals)
  # targets = jax.nn.one_hot(label, 2)

  preds = pred_graph.globals
  targets = label

  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  mask = jraph.get_graph_padding_mask(pred_graph)

  # MSE loss
  loss = jnp.mean(jnp.square((preds-targets)*mask[:,None]))

  # # Cross entropy loss.
  # loss = -jnp.mean(preds * targets * mask[:, None])

  # Accuracy taking into account the mask.
  # accuracy = jnp.sum(
  #     (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)
  return loss

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def train(dataset: List[GraphDataPoint], num_train_steps: int, mpn_steps:int) -> hk.Params:
  """Training loop."""

  net_fn_with_steps = partial(net_fn, steps=mpn_steps)
  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn_with_steps))
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0].input_graph

  # Initialize the network.
  params = net.init(jax.random.PRNGKey(42), graph)
  print('# of trainable parameters: ', count_params(params))
  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(1e-4)
  opt_state = opt_init(params)

  compute_loss_fn = functools.partial(compute_loss, net=net)
  # We jit the computation of our loss, since this is the main computation.
  # Using jax.jit means that we will use a single accelerator. If you want
  # to use more than 1 accelerator, use jax.pmap. More information can be
  # found in the jax documentation.
  compute_loss_fn = jax.jit(jax.value_and_grad(
      compute_loss_fn, has_aux=False))

  for idx in tqdm(range(num_train_steps)):
    graph = dataset[idx % len(dataset)].input_graph
    label = dataset[idx % len(dataset)].target
    # Jax will re-jit your graphnet every time a new graph shape is encountered.
    # In the limit, this means a new compilation every training step, which
    # will result in *extremely* slow training. To prevent this, pad each
    # batch of graphs to the nearest power of two. Since jax maintains a cache
    # of compiled programs, the compilation cost is amortized.
    graph = pad_graph_to_nearest_power_of_two(graph)

    # Since padding is implemented with pad_with_graphs, an extra graph has
    # been added to the batch, which means there should be an extra label.
    label = jnp.concatenate([label, jnp.array([0])])

    loss, grad = compute_loss_fn(params, graph, label)
    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    if idx % 50 == 0:
      print(f'step: {idx}, loss: {loss}')
  print('Training finished')
  return params

def evaluate(dataset: List[GraphDataPoint],
             params: hk.Params, mpn_steps:int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluation Script."""
  net_fn_with_steps = partial(net_fn, steps=mpn_steps)
  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn_with_steps))
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0].input_graph
  accumulated_loss = 0
  accumulated_accuracy = 0
  compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
  for idx in tqdm(range(len(dataset))):
    graph = dataset[idx].input_graph
    label = dataset[idx].target
    graph = pad_graph_to_nearest_power_of_two(graph)
    label = jnp.concatenate([label, jnp.array([0])])
    loss = compute_loss_fn(params, graph, label)
    # accumulated_accuracy += acc
    accumulated_loss += loss
    if idx % 100 == 0:
      print(f'Evaluated {idx + 1} graphs')
  print('Completed evaluation.')
  loss = accumulated_loss / idx
  # accuracy = accumulated_accuracy / idx
  # print(f'Eval loss: {loss}, accuracy {accuracy}')
  print(f'Eval loss: {loss}')
  return loss