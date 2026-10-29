#!/usr/bin/env python3
"""Produce the single-class ONNX detector from the pinned Google TF1 model.

Optional build-time dependencies: tensorflow-cpu==2.15.1 and tf2onnx==1.16.1.
This script is not needed to run the application. See public/models/README.md.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework.graph_util import extract_sub_graph

SOURCE_SHA256 = '9f49e034537b2518001794019869df0710f3089b0859ff10600408931a6a1895'
INPUT = 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3'
BOXES = 'Postprocessor/raw_box_locations'
SCORES = 'Postprocessor/raw_box_scores'
CLASS_COUNT = 603
HAMSTER_INDEX = 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_model', type=Path, help='original saved_model.pb')
    parser.add_argument('output', type=Path, help='output ONNX file')
    args = parser.parse_args()
    source = args.saved_model.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'Unexpected source SHA-256: {digest}')
    saved = saved_model_pb2.SavedModel()
    saved.ParseFromString(source)
    graph = saved.meta_graphs[0].graph_def
    labels = next(n for n in graph.node if n.name == 'hub_input/Const')
    if labels.attr['value'].tensor.string_val[HAMSTER_INDEX] != b'Hamster':
        raise ValueError('The expected model class is not Hamster')

    for node in graph.node:
        if node.name == INPUT:
            # The browser supplies the original model's pre-resized RGB pixels.
            node.Clear()
            node.name = INPUT
            node.op = 'Placeholder'
            node.attr['dtype'].type = tf.float32.as_datatype_enum
            node.attr['shape'].shape.CopyFrom(tf.TensorShape([1, 300, 300, 3]).as_proto())
        elif node.name.startswith('BoxPredictor_'):
            if node.name.endswith('/ClassPredictor/weights') or node.name.endswith('/ClassPredictor/biases'):
                values = tf.make_ndarray(node.attr['value'].tensor)
                if values.shape[-1] % CLASS_COUNT:
                    raise ValueError(f'Unexpected head shape: {node.name}')
                # Classes are interleaved per anchor; keep only Hamster.
                indices = np.arange(values.shape[-1] // CLASS_COUNT) * CLASS_COUNT + HAMSTER_INDEX
                node.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(values[..., indices]))
            elif node.name.endswith('/stack_1/2'):
                node.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(1, dtype=tf.int32))
        # Recalculate shapes after removing the other class channels.
        if '_output_shapes' in node.attr:
            del node.attr['_output_shapes']

    # Cut away the TF1 resize/control-flow and the 600-class NMS. Decode boxes
    # in the graph; select the best hamster and threshold in the application.
    graph = extract_sub_graph(graph, [BOXES, SCORES])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        graph_path = Path(temporary) / 'hamster.pb'
        graph_path.write_bytes(graph.SerializeToString())
        subprocess.run([
            sys.executable, '-m', 'tf2onnx.convert', '--graphdef', str(graph_path),
            '--inputs', f'{INPUT}:0[1,300,300,3]', '--outputs', f'{BOXES}:0,{SCORES}:0',
            '--opset', '13', '--output', str(args.output),
        ], check=True, env={**os.environ, 'PYTHONHASHSEED': '0'})
    print('ONNX SHA-256:', hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
