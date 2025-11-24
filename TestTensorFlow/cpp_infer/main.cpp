#include "facedetectcnn.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/c/c_api.h"

namespace{
    bool is_supported_image_file(const std::filesystem::path &path){
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char ch){
                           return static_cast<char>(std::tolower(ch));
                       });
        static const std::vector<std::string> kSupportedExtensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
        };
        return std::find(kSupportedExtensions.begin(), kSupportedExtensions.end(), ext) !=
               kSupportedExtensions.end();
    }

    void draw_face_rect(cv::Mat &image, const FaceRect &face){
        cv::rectangle(image,
                      cv::Rect(face.x, face.y, face.w, face.h),
                      cv::Scalar(0, 255, 0),
                      2);

        for(int i = 0; i < 5; ++ i){
            cv::circle(image,
                       cv::Point(face.lm[2 * i], face.lm[2 * i + 1]),
                       2,
                       cv::Scalar(0, 0, 255),
                       cv::FILLED);
        }
    }

    void CheckStatus(TF_Status *status, const std::string &msg){
        if(TF_GetCode(status) != TF_OK){
            std::cerr << msg << ": " << TF_Message(status) << std::endl;
            std::exit(1);
        }
    }

    TF_Status *g_tf_status = nullptr;
    TF_Graph *g_tf_graph = nullptr;
    TF_Session *g_tf_session = nullptr;
    TF_Output g_input_op;
    TF_Output g_output_op;

    void InitTensorFlow(const std::string &model_dir){
        g_tf_status = TF_NewStatus();

        TF_SessionOptions *sess_opts = TF_NewSessionOptions();
        TF_Buffer *run_opts = nullptr;

        const char *tags[] = {"serve"};
        int ntags = 1;

        g_tf_graph = TF_NewGraph();

        g_tf_session = TF_LoadSessionFromSavedModel(
                sess_opts,
                run_opts,
                model_dir.c_str(),
                tags,
                ntags,
                g_tf_graph,
                nullptr,
                g_tf_status
        );
        CheckStatus(g_tf_status, "Failed to load SavedModel");

        const char *input_op_name = "serving_default_keras_tensor";
        const char *output_op_name = "StatefulPartitionedCall_1";

        g_input_op.oper = TF_GraphOperationByName(g_tf_graph, input_op_name);
        g_input_op.index = 0;
        if(g_input_op.oper == nullptr){
            std::cerr << "Failed to find input op: " << input_op_name << std::endl;
            std::exit(1);
        }

        g_output_op.oper = TF_GraphOperationByName(g_tf_graph, output_op_name);
        g_output_op.index = 0;
        if(g_output_op.oper == nullptr){
            std::cerr << "Failed to find output op: " << output_op_name << std::endl;
            std::exit(1);
        }

        TF_DeleteSessionOptions(sess_opts);
    }

    void CleanupTensorFlow(){
        if(g_tf_session != nullptr){
            TF_DeleteSession(g_tf_session, g_tf_status);
        }
        if(g_tf_graph != nullptr){
            TF_DeleteGraph(g_tf_graph);
        }
        if(g_tf_status != nullptr){
            TF_DeleteStatus(g_tf_status);
        }
        g_tf_session = nullptr;
        g_tf_graph = nullptr;
        g_tf_status = nullptr;
    }

    TF_Tensor *MatToTensor(const cv::Mat &patch_bgr){
        cv::Mat resized_bgr;
        cv::resize(patch_bgr, resized_bgr, cv::Size(128, 128));

        cv::Mat patch_rgb;
        cv::cvtColor(resized_bgr, patch_rgb, cv::COLOR_BGR2RGB);

        int64_t dims[4] = {1, 128, 128, 3};
        size_t nbytes = 1 * 128 * 128 * 3 * sizeof(float);

        TF_Tensor *tensor = TF_AllocateTensor(TF_FLOAT, dims, 4, nbytes);
        float *data = static_cast<float *>(TF_TensorData(tensor));

        for(int y = 0; y < 128; ++ y){
            for(int x = 0; x < 128; ++ x){
                cv::Vec3b pixel = patch_rgb.at<cv::Vec3b>(y, x);
                int idx = (y * 128 + x) * 3;
                data[idx + 0] = static_cast<float>(pixel[0]);
                data[idx + 1] = static_cast<float>(pixel[1]);
                data[idx + 2] = static_cast<float>(pixel[2]);
            }
        }
        return tensor;
    }

    float ClassifyFacePatch(const cv::Mat &image, const FaceRect &face){
        int x = std::max(face.x, 0);
        int y = std::max(face.y, 0);
        int w = face.w;
        int h = face.h;

        if(x + w > image.cols){
            w = image.cols - x;
        }
        if(y + h > image.rows){
            h = image.rows - y;
        }

        if(w <= 0 || h <= 0){
            return 1.0f;
        }

        cv::Rect roi(x, y, w, h);
        cv::Mat patch_bgr = image(roi).clone();

        TF_Tensor *input_tensor = MatToTensor(patch_bgr);

        TF_Tensor *input_tensors[1] = {input_tensor};
        TF_Tensor *output_tensors[1] = {nullptr};

        TF_SessionRun(
                g_tf_session,
                nullptr,
                &g_input_op, input_tensors, 1,
                &g_output_op, output_tensors, 1,
                nullptr, 0,
                nullptr,
                g_tf_status
        );
        CheckStatus(g_tf_status, "Error during TF_SessionRun (face patch)");

        TF_Tensor *out = output_tensors[0];
        float *out_data = static_cast<float *>(TF_TensorData(out));
        float pred = out_data[0];

        TF_DeleteTensor(out);
        TF_DeleteTensor(input_tensor);

        return pred;
    }
}

/**
 * Main function
 * Working directory should be TestTensorFlow/cpp_infer
 * Images are read from ./input and results are saved to ./output
 */
int main(){
    const std::filesystem::path current_dir = std::filesystem::current_path();
    const std::filesystem::path input_dir = current_dir / "cpp_infer" / "input";
    const std::filesystem::path output_dir = current_dir / "cpp_infer" / "output";
    const std::filesystem::path model_dir =
            current_dir / "saved_models" / "face_classifier_savedmodel";

    std::cout << "Working directory: " << current_dir << std::endl;
    std::cout << "Using SavedModel directory: " << model_dir << std::endl;

    if(! std::filesystem::exists(model_dir)){
        std::cerr << "SavedModel directory does not exist: " << model_dir << std::endl;
        return 1;
    }
    if(! std::filesystem::exists(input_dir)){
        std::cerr << "Input directory does not exist: " << input_dir << std::endl;
        return 1;
    }

    std::filesystem::create_directories(output_dir);

    InitTensorFlow(model_dir.string());

    std::vector<std::filesystem::path> image_paths;
    for(const auto &entry: std::filesystem::directory_iterator(input_dir)){
        const auto &path = entry.path();
        if(entry.is_regular_file() && is_supported_image_file(path)){
            image_paths.push_back(path);
        }
    }

    if(image_paths.empty()){
        std::cout << "No supported images found in input directory: " << input_dir
                  << std::endl;
        CleanupTensorFlow();
        return 0;
    }

    const float kTfThreshold = 0.5f;

    for(const auto &image_path: image_paths){
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if(image.empty()){
            std::cerr << "Failed to read image: " << image_path << std::endl;
            continue;
        }

        std::cout << "Processing image: " << image_path << std::endl;

        std::vector<FaceRect> faces = objectdetect_cnn(image.data,
                                                       image.cols,
                                                       image.rows,
                                                       static_cast<int>(image.step));

        if(faces.empty()){
            std::cout << "No faces detected by libfacedetection." << std::endl;
        } else{
            std::cout << "Detected " << faces.size() << " faces by libfacedetection."
                      << std::endl;
        }

        size_t kept_count = 0;

        for(size_t idx = 0; idx < faces.size(); ++ idx){
            const FaceRect &face = faces[idx];

            float tf_pred = ClassifyFacePatch(image, face);

            std::cout << "  Face " << idx + 1
                      << ": libface_score=" << std::fixed << std::setprecision(3)
                      << face.score
                      << " tf_pred=" << tf_pred
                      << " bbox=[" << face.x << ", " << face.y
                      << ", " << face.w << ", " << face.h << "]";

            if(tf_pred < kTfThreshold){
                std::cout << " -> accepted by TF as FACE" << std::endl;
                draw_face_rect(image, face);
                ++ kept_count;
            } else{
                std::cout << " -> rejected by TF as NON-FACE" << std::endl;
            }
        }

        std::cout << "Final faces kept by TF: " << kept_count << std::endl;

        std::filesystem::path output_path =
                output_dir / (image_path.stem().string() + "_result" +
                              image_path.extension().string());
        if(! cv::imwrite(output_path.string(), image)){
            std::cerr << "Failed to save detection result: " << output_path << std::endl;
        } else{
            std::cout << "Result saved to: " << output_path << std::endl;
        }
    }

    CleanupTensorFlow();
    return 0;
}
