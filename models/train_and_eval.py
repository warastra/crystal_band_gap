import jax.numpy as jnp
import jraph
import haiku as hk
import jax
from typing import Tuple, List, Callable, Union, Dict
from utils.preprocessing import GraphDataPoint
from functools import partial
import optax
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from graph_model import count_params, pad_graph_to_nearest_power_of_two

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
#   and https://github.com/tisabe/jraph_MPEU/blob/master/jraph_MPEU/train.py
def compute_loss(
        params: hk.Params, 
        state:hk.State, 
        rng, 
        graph: jraph.GraphsTuple, 
        label: jnp.ndarray,
        net: jraph.GraphsTuple,
        is_eval:bool=False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
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
  
def train_eval_init(
        dataset: List[GraphDataPoint], 
        mpn_steps:int, 
        batch_size:int, 
        emb_size:int, 
        net_fn:Callable, 
        is_training:bool=False
    ) -> Dict:
    out_rng, init_rng = jax.random.split(jax.random.PRNGKey(42))
    net_fn_with_steps = partial(net_fn, steps=mpn_steps, emb_size=emb_size)
    # Transform impure `net_fn` to pure functions with hk.transform.
    net = hk.transform_with_state(net_fn_with_steps)
    # Get a candidate graph and label to initialize the network.

    if is_training:
        graph = jraph.batch([x.input_graph for x in dataset[0:batch_size]])
        params, hk_state = net.init(init_rng, graph)
        print('# of trainable parameters: ', count_params(params))
        # Initialize the optimizer.
        learning_rate = optax.exponential_decay(
            init_value=1e-4,
            transition_steps=50000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = optax.chain(optax.clip(0.2), optax.adamw(learning_rate=learning_rate))
        # opt_init, opt_update = optax.adam(1e-6)
        opt_state = optimizer.init(params)
        
        return {
            "params":params,
            "out_rng":out_rng,
            "optimizer":optimizer,
            "opt_state":opt_state,
            "hk_state":hk_state,
            "net":net
        }
    else:
        return {
            "net":net,
            "out_rng":out_rng,
        }
        
def train_step(
        dataset: List[GraphDataPoint],
        params:hk.Params,
        hk_state:hk.State,
        loss_with_grad_fn:Callable,
        net:Callable,
        optimizer,
        opt_state,
        step:int,
        batch_size:int,
        out_rng
    ):
    start_idx = (step*batch_size % len(dataset))
    end_idx = ((step+1)*batch_size % len(dataset))
    if end_idx < start_idx:
        graph = jraph.batch([x.input_graph for x in dataset[start_idx:]] + [x.input_graph for x in dataset[:end_idx]])
        label = jnp.array([x.target for x in dataset[start_idx:]] + [x.target for x in dataset[:end_idx]])
    else:
        graph = jraph.batch([x.input_graph for x in dataset[start_idx:end_idx]])
        label = jnp.array([x.target for x in dataset[start_idx:end_idx]])
    # Jax will re-jit your graphnet every time a new graph shape is encountered.
    # In the limit, this means a new compilation every training step, which
    # will result in *extremely* slow training. To prevent this, pad each
    # batch of graphs to the nearest power of two. Since jax maintains a cache
    # of compiled programs, the compilation cost is amortized.
    graph = pad_graph_to_nearest_power_of_two(graph)

    # Since padding is implemented with pad_with_graphs, an extra graph has
    # been added to the batch, which means there should be an extra label.
    label = jnp.concatenate([label, jnp.array([[0]])])

    loss, grad = loss_with_grad_fn(params, hk_state, out_rng, graph, label)
    _, updated_state = net.apply(params, hk_state, out_rng, graph)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    return updated_params, updated_state, opt_state, loss

def eval_step(
        dataset:List[GraphDataPoint], 
        net:Callable, 
        batch_size:int, 
        params:hk.Params, 
        hk_state:hk.State, 
        out_rng,
        loss_fn:Callable=compute_loss
    ):
    accumulated_loss = 0
    predictions = None

    compute_loss_fn = jax.jit(partial(loss_fn, net=net, is_eval=True))
    for idx in tqdm(range(len(dataset)//batch_size + 1)):
        graph = jraph.batch([x.input_graph for x in dataset[idx*batch_size:(idx+1)*batch_size]])
        label = jnp.array([x.target for x in dataset[idx*batch_size:(idx+1)*batch_size]])
        graph = pad_graph_to_nearest_power_of_two(graph)
        label = jnp.concatenate([label, jnp.array([[0]])])

        loss, batch_preds = compute_loss_fn(params, hk_state, out_rng, graph, label)
        # remove the last predictions because it's the prediction for the padded graph
        batch_preds = batch_preds[:-1] 
        if predictions is None:
            predictions = batch_preds
        else:
            predictions = jnp.concatenate([predictions, batch_preds])
        accumulated_loss += loss

    average_loss = accumulated_loss / len(dataset)
    return average_loss, predictions 

def train(
        train_dataset: List[GraphDataPoint], 
        val_dataset: List[GraphDataPoint], 
        num_train_steps: int, 
        mpn_steps:int, 
        batch_size:int, 
        emb_size:int, 
        net_fn:Callable
    ):
    """Training loop."""
    init_dict = train_eval_init(
                                dataset=train_dataset, 
                                mpn_steps=mpn_steps, 
                                batch_size=batch_size, 
                                emb_size=emb_size,
                                net_fn=net_fn,
                                is_training=True
                            )
    net = init_dict['net']
    out_rng = init_dict['out_rng']
    params = init_dict['params']
    hk_state = init_dict['hk_state']
    opt_state = init_dict['opt_state']
    optimizer = init_dict['optimizer']

    compute_loss_fn = partial(compute_loss, net=net)
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn_with_grad = jax.jit(jax.value_and_grad(compute_loss_fn))
    learning_curve = []
    for idx in tqdm(range(num_train_steps)):
        params, hk_state, opt_state, loss = train_step(
                                                dataset=train_dataset,
                                                params=params,
                                                hk_state=hk_state,
                                                loss_with_grad_fn=compute_loss_fn_with_grad,
                                                net=net,
                                                optimizer=optimizer,
                                                opt_state=opt_state,
                                                step=idx,
                                                batch_size=batch_size,
                                                out_rng=out_rng
                                            )
        if idx % 500 == 0 and idx != 0:
            print(f'step: {idx}, train loss: {loss}')
            validation_loss, val_preds = eval_step(
                                            val_dataset, 
                                            net=net, 
                                            batch_size=batch_size, 
                                            params=params, 
                                            hk_state=hk_state, 
                                            out_rng=out_rng, 
                                            loss_fn=compute_loss
                                        )
            y_val = [x.target for x in val_dataset]
            val_loss_sklearn = mean_squared_error(val_preds, y_val)
            print(f'step: {idx}, validation loss: {val_loss_sklearn}')

            learning_curve.append({
                "train_loss":loss,
                "validation_loss":validation_loss,
                "validation_loss_sklearn":val_loss_sklearn,
                "n_data_point":idx*batch_size
            })
    print('Training finished')
    return params, hk_state, learning_curve

def evaluate(
        dataset: List[GraphDataPoint],
        net_fn:Callable,
        params: hk.Params, 
        hk_state:hk.State,
        mpn_steps:int,
        batch_size:int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation Script."""
    init_dict = train_eval_init(dataset=dataset, 
                                mpn_steps=mpn_steps, 
                                batch_size=batch_size, 
                                emb_size=params['linear_1']['w'].shape[1],
                                net_fn=net_fn,
                                is_training=False
                            )
    net = init_dict['net']
    out_rng = init_dict['out_rng']

    loss, predictions = eval_step(
            dataset=dataset, 
            net=net, 
            batch_size=batch_size, 
            params=params, 
            hk_state=hk_state, 
            out_rng=out_rng,
            loss_fn=compute_loss
        )
    
    print('Completed evaluation.')
    print(f'Eval loss: {loss}')

    return loss, predictions