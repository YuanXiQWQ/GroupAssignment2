# GroupAssignment2

[EN](README.md) | **ZH**

BCIT's CST COMP3940 小组作业 2 (ML & AI 组)

## 规划

### 11 月 19 日截止：[4 分] 个人演示（线上或线下）

- **Persphone**：在 Linux 上安装并配置 OpenCV，获取摄像头画面以捕捉图片与视频。
- **Lin**：安装并训练 TensorFlow（C++），用于识别小规模人脸（如团队成员）。
- **Jerry**：安装并训练 TensorFlow（C++），实现多脸检测。
- **Yang**：尝试使用 C++ 从零实现的人脸识别方案：

  [libfacedetection](https://github.com/ShiqiYu/libfacedetection) |
  [kirill-menke/neural-network-optimization](https://github.com/kirill-menke/neural-network-optimization)

- **Shawn**：安装并配置 Whisper（C++），实现语音识别。
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp)

- **Clinton**：安装并配置 VOSK，实现语音与说话人识别。
- **Kevin**：在 libtorch 中安装并配置 Wav2Vec2，实现语音与说话人识别。

### 11 月 26 日截止：[4 分] 个人演示（线上或线下）

- **Persphone、Lin、Jerry 与 Yang**：整合 OpenCV 与 TensorFlow，开发视频流人脸检测方案。
- **Shawn、Clinton 与 Kevin**：甄选最佳方案，实现实时对话的语音转写并标识说话人。

### 12 月 1 日截止：[2 分] 最终综合演示

- 展示多脸识别与具备说话人识别功能的实时转写。

## 许可证

本项目使用 [MIT 许可证](LICENSE) 进行授权。

本项目依赖以下第三方库：

- [Libfacedetection](https://github.com/ShiqiYu/libfacedetection)
  使用 [3-clause BSD 许可证](https://github.com/ShiqiYu/libfacedetection/blob/master/LICENSE)。根据其许可证要求，您在分发本项目时：
    - 必须保留 libfacedetection 源码或二进制中包含的**原始版权声明**。

- [TensorFlow](https://github.com/tensorflow/tensorflow)
  使用 [Apache 许可证 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)。
    - 在分发本项目时应当在文档中**标注来源**。
    - 若您修改了 TensorFlow 源码，必须在文档中**声明您做过修改**。

- [OpenCV](https://github.com/opencv/opencv)
  同样使用 [Apache 许可证 2.0](https://github.com/opencv/opencv/blob/master/LICENSE)，要求与 TensorFlow 相同。

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp), 使用
  [MIT 许可证](https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE).