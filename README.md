# GroupAssignment2

**EN** | [ZH](README_zh.md)

BCIT's CST COMP3940 Group Assignment 2 (Group ML & AI)

## Schedule

### Due Nov 19th :  [4 marks] Individual Demos online or in-person

- **Persphone**:  Install & Configure OpenCV on Linux to gain control of your webcam to grab pictures and video.
- **Lin**:  Install and train TensorFlow (C++) for face recognition of small set of faces (e.g. team members).
- **Jerry**: Install and train TensorFlow (C++) for multiface detection.
- **Yang**: Try out implementations from scratch in C++ in face recognition:

  [libfacedetection](https://github.com/ShiqiYu/libfacedetection) |
  [kirill-menke/neural-network-optimization](https://github.com/kirill-menke/neural-network-optimization)

- **Shawn**: Install & configure Whisper (C++) for speech recognition
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp)

- **Clinton**:  Install and configure VOSK for speech and speaker identification.
- **Kevin**: install and configure Wav2Vec2 in libtorch for speech and speaker identification.

### Due Nov 26th :  [4 marks] Individual Demos online or in-person

- **Persphone, Lin, Jerry & Yang**:  Integrating OpenCV with TensorFlow, develop a solution for detecting faces in video
stream
- **Shawn, Clinton & Kevin**: Choose the best solution that can transcribe a live conversation identifying speakers

### Due Dec 1st [2 marks]  Final Integrated Demo

- Demonstrate MultiFace Identification and Transcribing with Speaker Identification.

## License

This project is distributed under the [MIT License](LICENSE).

This project depends on the following third-party libraries:

- [libfacedetection](https://github.com/ShiqiYu/libfacedetection), licensed under the
  [3-clause BSD License](https://github.com/ShiqiYu/libfacedetection/blob/master/LICENSE). When redistributing this
  project, you must:
  - **Retain the original copyright notice** included in libfacedetection’s source or binary files.

- [TensorFlow](https://github.com/tensorflow/tensorflow), licensed under the
  [Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).
    - When redistributing the project, you should **credit TensorFlow** in your documentation.
    - If you modify TensorFlow’s source code, you must **document the changes you made**.

- [OpenCV](https://github.com/opencv/opencv), also licensed under the
  [Apache License 2.0](https://github.com/opencv/opencv/blob/master/LICENSE), with the same requirements as TensorFlow.

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp), licensed under the
  [MIT License](https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE).