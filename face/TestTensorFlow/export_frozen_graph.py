import pathlib

import tensorflow as tf
from tensorflow import keras

try:
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )
except ImportError as exc:  # pragma: no cover - depends on TF build
    raise ImportError(
        "TensorFlow build missing convert_variables_to_constants_v2; "
        "please upgrade TensorFlow."
    ) from exc


def main() -> None:
    base_dir = pathlib.Path(__file__).resolve().parent
    saved_models_dir = base_dir / "saved_models"

    keras_path = saved_models_dir / "face_classifier.keras"
    if not keras_path.exists():
        raise FileNotFoundError(f"Missing Keras model file: {keras_path}")

    pb_path = saved_models_dir / "face_classifier_frozen.pb"
    pbtxt_path = saved_models_dir / "face_classifier_frozen.pbtxt"

    model = keras.models.load_model(keras_path)

    # Build a concrete function for inference
    input_tensor = model.inputs[0]
    input_name = input_tensor.name.split(":")[0]

    concrete_func = tf.function(model).get_concrete_function(
        tf.TensorSpec(
            shape=input_tensor.shape,
            dtype=input_tensor.dtype,
            name=input_name,
        )
    )

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.io.gfile.GFile(pb_path, "wb") as f:
        f.write(graph_def.SerializeToString())

    tf.io.write_graph(
        graph_def,
        str(saved_models_dir),
        pbtxt_path.name,
        as_text=True,
    )

    print(f"Frozen graph saved to: {pb_path}")
    print(f"Frozen graph (text) saved to: {pbtxt_path}")

    print("Inputs:")
    for tensor in frozen_func.inputs:
        print(f"  {tensor.name} -> shape {tensor.shape}, dtype {tensor.dtype}")

    print("Outputs:")
    for tensor in frozen_func.outputs:
        print(f"  {tensor.name} -> shape {tensor.shape}, dtype {tensor.dtype}")


if __name__ == "__main__":
    main()
