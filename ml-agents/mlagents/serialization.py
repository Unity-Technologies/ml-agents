import copy
import logging
from typing import Any, List, Set, Tuple, NamedTuple, Iterable

import onnx

from tf2onnx.tfonnx import process_tf_graph, tf_optimize
from tf2onnx import optimizer


from mlagents.tf_utils import tf
from mlagents.trainers.tf_policy import TFPolicy

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from mlagents.trainers import tensorflow_to_barracuda as tf2bc

logger = logging.getLogger("mlagents.trainers")

POSSIBLE_INPUT_NODES = frozenset(
    [
        "action_masks",
        "epsilon",
        "prev_action",
        "recurrent_in",
        "sequence_length",
        "vector_observation",
    ]
)

POSSIBLE_OUTPUT_NODES = frozenset(
    ["action", "action_probs", "recurrent_out", "value_estimate"]
)

MODEL_CONSTANTS = frozenset(
    ["action_output_shape", "is_continuous_control", "memory_size", "version_number"]
)
VISUAL_OBSERVATION_PREFIX = "visual_observation_"


class SerializationSettings(NamedTuple):
    model_path: str
    brain_name: str
    convert_to_barracuda: bool = True
    convert_to_onnx: bool = True
    onnx_opset: int = 8


def export_policy_model(settings: SerializationSettings, policy: TFPolicy) -> None:
    """
    Exports latest saved model to .nn format for Unity embedding.
    """
    frozen_graph_def = _make_frozen_graph(settings, policy)
    # Save frozen graph and convert to barracuda
    frozen_graph_def_path = settings.model_path + "/frozen_graph_def.pb"
    with gfile.GFile(frozen_graph_def_path, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

    if settings.convert_to_barracuda:
        tf2bc.convert(frozen_graph_def_path, settings.model_path + ".nn")
        logger.info(f"Exported {settings.model_path}.nn file")

    # Save to onnx too
    if settings.convert_to_onnx:
        onnx_graph = convert_frozen_to_onnx(settings, frozen_graph_def)
        onnx_output_path = settings.model_path + "_onnx.onnx"
        with open(onnx_output_path, "wb") as f:
            f.write(onnx_graph.SerializeToString())
        logger.info(f"Converting to {onnx_output_path}")


def _make_frozen_graph(
    settings: SerializationSettings, policy: TFPolicy
) -> tf.GraphDef:
    with policy.graph.as_default():
        target_nodes = ",".join(_process_graph(settings, policy))
        graph_def = policy.graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            policy.sess, graph_def, target_nodes.replace(" ", "").split(",")
        )
    return output_graph_def


def convert_frozen_to_onnx(
    settings: SerializationSettings, frozen_graph_def: tf.GraphDef
) -> onnx.ModelProto:
    # This is basically https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/convert.py

    # Some constants in the graph need to be read by the inference system.
    # These aren't used by the model anywhere, so trying to make sure they propagate
    # through conversion and import is a losing battle. Instead, save them now,
    # so that we can add them back later.
    constant_values = {}
    for n in frozen_graph_def.node:
        if n.name in MODEL_CONSTANTS:
            val = n.attr["value"].tensor.int_val[0]
            constant_values[n.name] = val

    # TODO set this if --debug is set
    # logging.basicConfig(level=logging.get_verbosity_level(1))

    inputs, vis_inputs = _get_input_node_names(frozen_graph_def)
    outputs = _get_output_node_names(frozen_graph_def)
    logger.info(f"onnx export - inputs:{inputs} outputs:{outputs}")

    frozen_graph_def = _fixup_conv_transposes(vis_inputs, frozen_graph_def)

    frozen_graph_def = tf_optimize(
        inputs, outputs, frozen_graph_def, fold_constant=True
    )

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(frozen_graph_def, name="")
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(
            tf_graph,
            input_names=inputs,
            # inputs_as_nchw=vis_inputs,
            output_names=outputs,
            opset=settings.onnx_opset,
        )

    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model(settings.brain_name)

    # Save the constant values back the graph initializer.
    # This will ensure the importer gets them as global constants.
    constant_nodes = []
    for k, v in constant_values.items():
        constant_node = _make_onnx_node_for_constant(k, v)
        constant_nodes.append(constant_node)
    model_proto.graph.initializer.extend(constant_nodes)
    return model_proto


def _make_onnx_node_for_constant(name: str, value: int) -> Any:
    tensor_value = onnx.TensorProto(
        data_type=onnx.TensorProto.INT32,
        name=name,
        int32_data=[value],
        dims=[1, 1, 1, 1],
    )
    return tensor_value


def _get_input_node_names(frozen_graph_def: Any) -> Tuple[List[str], List[str]]:
    node_names = _get_frozen_graph_node_names(frozen_graph_def)
    input_names = node_names & POSSIBLE_INPUT_NODES

    # Check visual inputs sequentially, and exit as soon as we don't find one
    vis_index = 0
    vis_obs_names = set()
    while True:
        vis_node_name = f"{VISUAL_OBSERVATION_PREFIX}{vis_index}"
        if vis_node_name in node_names:
            input_names.add(vis_node_name)
            vis_obs_names.add(vis_node_name)
        else:
            break
        vis_index += 1
    # Append the port
    return [f"{n}:0" for n in input_names], [f"{n}:0" for n in vis_obs_names]


def _get_output_node_names(frozen_graph_def: Any) -> List[str]:
    node_names = _get_frozen_graph_node_names(frozen_graph_def)
    output_names = node_names & POSSIBLE_OUTPUT_NODES
    # Append the port
    return [f"{n}:0" for n in output_names]


def _get_frozen_graph_node_names(frozen_graph_def: Any) -> Set[str]:
    names = set()
    for node in frozen_graph_def.node:
        names.add(node.name)
    return names


def _fixup_conv_transposes(
    vis_input_names: Iterable[str], frozen_graph: tf.GraphDef
) -> tf.GraphDef:
    # create new model and replace all data_format
    new_graph = tf.GraphDef()
    for n in frozen_graph.node:
        nn = new_graph.node.add()
        nn.CopyFrom(n)

    # remove :0 from the import names
    vis_input_names = {n.rsplit(":")[0] for n in vis_input_names}

    previous_op = ""
    for node in new_graph.node:
        if node.name in vis_input_names:
            print(f"permuting dimensions for {node.name}")
            node_dim = copy.deepcopy(node.attr["shape"])
            dim = len(node_dim.shape.dim)
            if dim == 4:
                node.attr["shape"].shape.dim[0].size = (
                    node_dim.shape.dim[0].size if dim > 0 else -1
                )
                node.attr["shape"].shape.dim[1].size = (
                    node_dim.shape.dim[3].size if dim > 3 else -1
                )
                node.attr["shape"].shape.dim[2].size = (
                    node_dim.shape.dim[1].size if dim > 1 else -1
                )
                node.attr["shape"].shape.dim[3].size = (
                    node_dim.shape.dim[2].size if dim > 2 else -1
                )

        if node.op == "Conv2D":
            node.attr["data_format"].s = str.encode("NCHW")
            strides = copy.deepcopy(node.attr["strides"])
            node.attr["strides"].list.i[0] = strides.list.i[0]
            node.attr["strides"].list.i[1] = strides.list.i[3]
            node.attr["strides"].list.i[2] = strides.list.i[1]
            node.attr["strides"].list.i[3] = strides.list.i[2]

        # ONNX, bias is merged into Conv2D, so need to change it's layout
        if node.op == "Conv2D" or (node.op == "BiasAdd" and previous_op == "Conv2D"):
            print(f"setting {node.name} to NCHW")
            node.attr["data_format"].s = str.encode("NCHW")

        if node.op == "Conv1D" or (node.op == "BiasAdd" and previous_op == "Conv1D"):
            print(f"setting {node.name} to NCHW")
            node.attr["data_format"].s = str.encode("NCHW")

        previous_op = node.op
    return new_graph


def _process_graph(settings: SerializationSettings, policy: TFPolicy) -> List[str]:
    """
    Gets the list of the output nodes present in the graph for inference
    :return: list of node names
    """
    all_nodes = [x.name for x in policy.graph.as_graph_def().node]
    nodes = [x for x in all_nodes if x in POSSIBLE_OUTPUT_NODES | MODEL_CONSTANTS]
    logger.info("List of nodes to export for brain :" + settings.brain_name)
    for n in nodes:
        logger.info("\t" + n)
    return nodes
