import pathlib

import tensorflow as tf
from tensorflow import keras

try:
    from tensorflow.python.framework.convert_to_constants import (  # type: ignore[attr-defined]
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

    savedmodel_dir = saved_models_dir / "face_classifier_savedmodel"
    if not savedmodel_dir.exists():
        raise FileNotFoundError(
            "Missing SavedModel export. Please re-run training or export SavedModel."
        )

    pb_path = saved_models_dir / "face_classifier_frozen.pb"
    pbtxt_path = saved_models_dir / "face_classifier_frozen.pbtxt"

    # Load SavedModel signature to ensure variables are folded for inference
    loaded = tf.saved_model.load(savedmodel_dir)
    if "serving_default" not in loaded.signatures:
        raise KeyError("SavedModel missing 'serving_default' signature")

    concrete_func = loaded.signatures["serving_default"]

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
