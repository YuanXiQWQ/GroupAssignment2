# GroupAssignment2

**English** | [中文](#中文)

BCIT's CST COMP3940 Group Assignment 2 (Group ML & AI)

# English

## TestOpenCV

This project is actually used to test OpenCV and verify [libfacedetection](https://github.com/ShiqiYu/libfacedetection).
Model training and deployment are not included here.

## TestTensorFlow

- The root of this directory is used for training models, primarily in Python. The `data` folder is ignored by version
  control. If you want to train the model yourself with this project’s code, create the following structure under
  `TestTensorFlow`:

```
data
|__train          <-- Training data; to keep the model stable, prepare at least 500 samples
    |__face       <-- Images containing faces
    |__non_face   <-- Images without faces; should be roughly a 1:1 ratio with face images
|__val            <-- Validation data; the number of images should be roughly 4:1 compared to train
    |__face
    |__non_face
```

- You can use `split_train_val.py` to automatically split 20% of the images from `train` into `val`, so you don’t have
  to move them manually.

- Run `train_face_classifier.py` to train the model. The trained models are saved under `saved_models`.

- The actual C++ implementation is located in `cpp_infer`. To run tests, load the CMake project under `cpp_infer` and
  set the working directory to `TestTensorFlow`. Place the images you want to classify in `cpp_infer/input`. If the run
  succeeds, you should see the results in `cpp_infer/output`.

## License

This project is distributed under the [MIT](LICENSE) license.

This project depends on the following third-party libraries:

- [libfacedetection](https://github.com/ShiqiYu/libfacedetection), licensed under the
  [3-clause BSD License](https://github.com/ShiqiYu/libfacedetection/blob/master/LICENSE). When redistributing this
  project, you must **retain the original copyright notice** included in libfacedetection’s source or binary files.

- [TensorFlow](https://github.com/tensorflow/tensorflow), licensed under the
  [Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).
    - When redistributing the project, you should **credit TensorFlow** in your documentation.
    - If you modify TensorFlow’s source code, you must **document the changes you made**.

- [OpenCV](https://github.com/opencv/opencv), also licensed under the
  [Apache License 2.0](https://github.com/opencv/opencv/blob/master/LICENSE), with the same requirements as TensorFlow.

---

[English](#English) | **中文**

BCIT's CST COMP3940 小组作业 2 (ML & AI 组)

# 中文

## TestOpenCV

实际上是用来测试OpenCV和验证[libfacedetection](https://github.com/ShiqiYu/libfacedetection)的。模型的训练和使用不在这里。

## TestTensorFlow

- 该文件的根目录用来训练模型，主要是 Python。已经忽略了 `data` 的提交，如果您希望使用该项目的代码自行训练模型，您需要在
  `TestTensorFlow` 创建如下结构的目录：

```
`data`
|__`train`          <-- 用于训练的数据，若要确保模型稳定，数量至少在 500 以上
    |__`face`       <-- 包含脸部的图片
    |__`non_face`   <-- 不包含脸部的图片，与包含脸部的图片数量应大致为1:1
|__`val`            <-- 用于验证的数据，与训练的数据的图片数量应大致为4:1
    |__`face`
    |__`non_face`
```

- 您可以使用 `split_train_val.py` 来从 `train` 自动分割 20% 的图片到`val`中，不必手动分别放置图片。

- 通过执行 `train_face_classifier.py` 来训练模型。模型的输出位置在`saved_models`。

- 实际的 C++ 实现位于 `cpp_infer`。若要测试，需要加载`cpp_infer`下的CMake项目，并将工作目录设置为`TestTensorFlow`
  。将您要识别的图片放在
  `cpp_infer/input`内，若执行成功，您应该在`cpp_infer/output`看到结果。

## 许可证

本项目使用 [MIT](LICENSE) 许可证进行授权。

本项目依赖以下第三方库：

- [Libfacedetection](https://github.com/ShiqiYu/libfacedetection)
  使用 [3-clause BSD License](https://github.com/ShiqiYu/libfacedetection/blob/master/LICENSE)。根据其许可证要求，您在分发本项目时：
    - 必须保留 libfacedetection 源码或二进制中包含的**原始版权声明**。

- [TensorFlow](https://github.com/tensorflow/tensorflow)
  使用 [Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)。
    - 在分发本项目时应当在文档中**标注来源**。
    - 若您修改了 TensorFlow 源码，必须在文档中**声明您做过修改**。

- [OpenCV](https://github.com/opencv/opencv)
  同样使用 [Apache License 2.0](https://github.com/opencv/opencv/blob/master/LICENSE)，要求与 TensorFlow 相同。
