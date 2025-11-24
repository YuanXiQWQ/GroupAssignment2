# face

有关人脸检测/识别的组件。

[EN](README.md) | **ZH**

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

- 您可以使用 `download_pubfig_faces` 来从
  [PubFig](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/)
  数据集下载面部图片，但是会很慢且效率很低，质量也不高（这个数据集很老了）。推荐直接从
  [Kaggle 数据集](https://www.kaggle.com/datasets/sagarkarar/nonface-and-face-dataset?resource=download)
  下载图片后直接放到 `train` 目录。其它数据集我还没找过。

- 您可以使用 `split_train_val.py` 来从 `train` 自动分割 20% 的图片到 `val` 中，不必手动分别放置图片。

- 通过执行 `train_face_classifier.py` 来训练模型。模型的输出位置在`saved_models`。

- 实际的 C++ 实现位于 `cpp_infer`。若要测试，需要加载 `cpp_infer` 下的 CMake 项目，并将工作目录设置为 `TestTensorFlow`
  。将您要识别的图片放在
  `cpp_infer/input` 内，若执行成功，您应该在 `cpp_infer/output` 看到结果。