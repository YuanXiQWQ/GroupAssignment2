# face

Components related to face detection/recognition.

**EN** | [ZH](README_zh.md)

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

- You can use `download_pubfig_faces.py` to fetch face images from
  the [PubFig dataset](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/),
  but it is slow, inefficient, and the image quality is dated. It’s recommended to download images directly from
  [Kaggle dataset](https://www.kaggle.com/datasets/sagarkarar/nonface-and-face-dataset?resource=download) and place
  them under `train` instead. I haven't found any other datasets yet.

- You can use `split_train_val.py` to automatically split 20% of the images from `train` into `val`, so you don’t have
  to move them manually.

- Run `train_face_classifier.py` to train the model. The trained models are saved under `saved_models`.

- The actual C++ implementation is located in `cpp_infer`. To run tests, load the CMake project under `cpp_infer` and
  set the working directory to `TestTensorFlow`. Place the images you want to classify in `cpp_infer/input`. If the run
  succeeds, you should see the results in `cpp_infer/output`.
