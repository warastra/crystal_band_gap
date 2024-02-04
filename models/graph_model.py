import jax.numpy as jnp
from jraph import GraphsTuple
import jraph
import haiku as hk
import jax
from jax.nn import initializers
from typing import Tuple, List, Dict, Any, Sequence
from utils.preprocessing import GraphDataPoint
from functools import partial
import optax
from gnome_model.crystal import mlp
from sklearn.metrics import mean_squared_error, mean_squared_log_error

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

def count_params(params:hk.Params):
  w_params = 0
  b_params = 0
  total_params = 0
  for layer in params.keys():
      try:
        w_size = params[layer]['w'].shape
        w_param_cnt = w_size[0] * w_size[1]
        b_param_count = params[layer]['b'].shape[0]

        w_params += w_param_cnt
        b_params += b_param_count
        total_params += w_param_cnt + b_param_count
      except:
        try:
          l_size = params[layer]['scale'].shape
        except:
          l_size = params[layer].shape
        l_param_cnt = l_size[0]
        total_params += l_param_cnt

  return total_params, w_params, b_params

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def base_mlp(feats: jnp.ndarray, emb_size:int=128, activation_fn=jax.nn.silu, w_init_fn=None, layername:str=None) -> jnp.ndarray:
  """to be used as update functions for graph net."""
  net = hk.Sequential(
        [
          hk.Linear(emb_size, w_init=w_init_fn), activation_fn,
          hk.Linear(emb_size, w_init=w_init_fn), 
          hk.LayerNorm(axis=-1,
                  create_scale=True,
                  create_offset=True)
        ]
        , name=layername
      )
  return net(feats)

def readout_mlp(feats: jnp.ndarray, emb_size:int=128, activation_fn=jax.nn.silu, w_init_fn=None) -> jnp.ndarray:
  """to be used as update functions for graph net."""
  # BandGap Prediction is a regression, so output a single value.
  net = hk.Sequential(
      [
        hk.Linear(emb_size, w_init=w_init_fn), activation_fn, 
        hk.LayerNorm(axis=-1,
                  create_scale=True,
                  create_offset=True,),
        hk.Linear(1)
      ] 
      , name='global_readout_linear')
  return net(feats)

def net_fn(graph: jraph.GraphsTuple, steps:int, emb_size:int=128) -> jraph.GraphsTuple:
  # Add a global paramater for graph classification.
  graph = graph._replace(globals=jnp.ones([graph.n_node.shape[0], 1]))
  embedder = jraph.GraphMapFeatures(
      hk.Linear(emb_size), 
      hk.Linear(emb_size), 
      hk.Linear(emb_size))
  
  weight_init_fn = hk.initializers.TruncatedNormal(1. / np.sqrt(1e4))
  mlp_fn = partial(base_mlp, emb_size=emb_size, activation_fn=jax.nn.silu, w_init_fn=weight_init_fn)
  readout_global_fn = partial(readout_mlp, emb_size=emb_size, activation_fn=jax.nn.silu, w_init_fn=weight_init_fn)
  node_update_fn = update_edge_fn = update_global_fn = mlp_fn

  
  output = embedder(graph)
  for i in range(steps):
    net = jraph.GraphNetwork(
        update_node_fn=jraph.concatenated_args(partial(node_update_fn, layername='mpnn_node_step_{}'.format(str(i)))),
        update_edge_fn=jraph.concatenated_args(partial(update_edge_fn, layername='mpnn_edge_step_{}'.format(str(i)))),
        update_global_fn=jraph.concatenated_args(partial(update_global_fn, layername='mpnn_global_step_{}'.format(str(i))))
      )
    output = net(output)

  readout = jraph.GraphNetwork(
      update_node_fn=jraph.concatenated_args(partial(node_update_fn, layername='node_linear_readout')),
      update_edge_fn=jraph.concatenated_args(partial(update_edge_fn, layername='edge_linear_readout')),
      update_global_fn=jraph.concatenated_args(readout_global_fn))
  return readout(output)
  
def compute_loss(params: hk.Params, 
                 state:hk.State, 
                 rng, 
                 graph: jraph.GraphsTuple, 
                 label: jnp.ndarray,
                 net: jraph.GraphsTuple,
                 is_eval:bool=False
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes loss and accuracy."""
  pred_graph, new_state = net.apply(params, state, rng, graph)
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
  squared_diff = jnp.square((preds-targets)*mask[:,None])
  mean_loss = jnp.sum(squared_diff) / jnp.sum(mask)

  # # Cross entropy loss.
  # loss = -jnp.mean(preds * targets * mask[:, None])

  # Accuracy taking into account the mask.
  # accuracy = jnp.sum(
  #     (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)
  if is_eval:
    return mean_loss, preds
  else:
    return mean_loss
