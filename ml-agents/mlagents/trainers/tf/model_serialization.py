from distutils.util import strtobool
import os
from typing import Any, List, Set
from distutils.version import LooseVersion

try:
    from tf2onnx.tfonnx import process_tf_graph, tf_optimize
    from tf2onnx import optimizer

    ONNX_EXPORT_ENABLED = True
except ImportError:
    # Either onnx and tf2onnx not installed, or they're not compatible with the version of tensorflow
    ONNX_EXPORT_ENABLED = False
    pass

from mlagents.tf_utils import tf

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import SerializationSettings
from mlagents.trainers.tf import tensorflow_to_barracuda as tf2bc

if LooseVersion(tf.__version__) < LooseVersion("1.12.0"):
    # ONNX is only tested on 1.12.0 and later
    ONNX_EXPORT_ENABLED = False

logger = get_logger(__name__)


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

POSSIBLE_OUTPUT_NODES = frozenset(["action", "recurrent_out", "value_estimate"])

MODEL_CONSTANTS = frozenset(
    [
        "action_output_shape",
        "is_continuous_control",
        "memory_size",
        "version_number",
        "trainer_major_version",
        "trainer_minor_version",
        "trainer_patch_version",
    ]
)
VISUAL_OBSERVATION_PREFIX = "visual_observation_"


def export_policy_model(
    model_path: str,
    output_filepath: str,
    behavior_name: str,
    graph: tf.Graph,
    sess: tf.Session,
) -> None:
    """
    Exports a TF graph for a Policy to .nn and/or .onnx format for Unity embedding.

    :param output_filepath: file path to output the model (without file suffix)
    :param behavior_name: behavior name of the trained model
    :param graph: Tensorflow Graph for the policy
    :param sess: Tensorflow session for the policy
    """
    frozen_graph_def = _make_frozen_graph(behavior_name, graph, sess)
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    # Save frozen graph
    frozen_graph_def_path = model_path + "/frozen_graph_def.pb"
    with gfile.GFile(frozen_graph_def_path, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

    # Convert to barracuda
    if SerializationSettings.convert_to_barracuda:
        tf2bc.convert(frozen_graph_def_path, f"{output_filepath}.nn")
        logger.info(f"Exported {output_filepath}.nn")

    # Save to onnx too (if we were able to import it)
    if ONNX_EXPORT_ENABLED:
        if SerializationSettings.convert_to_onnx:
            try:
                onnx_graph = convert_frozen_to_onnx(behavior_name, frozen_graph_def)
                onnx_output_path = f"{output_filepath}.onnx"
                with open(onnx_output_path, "wb") as f:
                    f.write(onnx_graph.SerializeToString())
                logger.info(f"Converting to {onnx_output_path}")
            except Exception:
                # Make conversion errors fatal depending on environment variables (only done during CI)
                if _enforce_onnx_conversion():
                    raise
                logger.exception(
                    "Exception trying to save ONNX graph. Please report this error on "
                    "https://github.com/Unity-Technologies/ml-agents/issues and "
                    "attach a copy of frozen_graph_def.pb"
                )

    else:
        if _enforce_onnx_conversion():
            raise RuntimeError(
                "ONNX conversion enforced, but couldn't import dependencies."
            )


def _make_frozen_graph(
    behavior_name: str, graph: tf.Graph, sess: tf.Session
) -> tf.GraphDef:
    with graph.as_default():
        target_nodes = ",".join(_process_graph(behavior_name, graph))
        graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph_def, target_nodes.replace(" ", "").split(",")
        )
    return output_graph_def


def convert_frozen_to_onnx(behavior_name: str, frozen_graph_def: tf.GraphDef) -> Any:
    # This is basically https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/convert.py

    inputs = _get_input_node_names(frozen_graph_def)
    outputs = _get_output_node_names(frozen_graph_def)
    logger.info(f"onnx export - inputs:{inputs} outputs:{outputs}")

    frozen_graph_def = tf_optimize(
        inputs, outputs, frozen_graph_def, fold_constant=True
    )

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(frozen_graph_def, name="")
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(
            tf_graph,
            input_names=inputs,
            output_names=outputs,
            opset=SerializationSettings.onnx_opset,
        )

    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model(behavior_name)

    return model_proto


def _get_input_node_names(frozen_graph_def: Any) -> List[str]:
    """
    Get the list of input node names from the graph.
    Names are suffixed with ":0"
    """
    node_names = _get_frozen_graph_node_names(frozen_graph_def)
    input_names = node_names & POSSIBLE_INPUT_NODES

    # Check visual inputs sequentially, and exit as soon as we don't find one
    vis_index = 0
    while True:
        vis_node_name = f"{VISUAL_OBSERVATION_PREFIX}{vis_index}"
        if vis_node_name in node_names:
            input_names.add(vis_node_name)
        else:
            break
        vis_index += 1
    # Append the port
    return [f"{n}:0" for n in input_names]


def _get_output_node_names(frozen_graph_def: Any) -> List[str]:
    """
    Get the list of output node names from the graph.
    Also include constants, so that they will be readable by the
    onnx importer.
    Names are suffixed with ":0"
    """
    node_names = _get_frozen_graph_node_names(frozen_graph_def)
    output_names = node_names & (POSSIBLE_OUTPUT_NODES | MODEL_CONSTANTS)
    # Append the port
    return [f"{n}:0" for n in output_names]


def _get_frozen_graph_node_names(frozen_graph_def: Any) -> Set[str]:
    """
    Get all the node names from the graph.
    """
    names = set()
    for node in frozen_graph_def.node:
        names.add(node.name)
    return names


def _process_graph(behavior_name: str, graph: tf.Graph) -> List[str]:
    """
    Gets the list of the output nodes present in the graph for inference
    :return: list of node names
    """
    all_nodes = [x.name for x in graph.as_graph_def().node]
    nodes = [x for x in all_nodes if x in POSSIBLE_OUTPUT_NODES | MODEL_CONSTANTS]
    logger.info("List of nodes to export for behavior :" + behavior_name)
    for n in nodes:
        logger.info("\t" + n)
    return nodes


def _enforce_onnx_conversion() -> bool:
    env_var_name = "TEST_ENFORCE_ONNX_CONVERSION"
    if env_var_name not in os.environ:
        return False

    val = os.environ[env_var_name]
    try:
        # This handles e.g. "false" converting reasonably to False
        return strtobool(val)
    except Exception:
        return False
