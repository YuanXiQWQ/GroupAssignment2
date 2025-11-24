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

/**
 * Helper functions
 */
namespace{
    /**
     * Check if the path contains the "outputs" component
     * @param path The path to inspect for an "outputs" directory component
     * @return true if any segment of the path equals "outputs"
     */
    bool contains_outputs_component(const std::filesystem::path &path){
        return std::any_of(path.begin(), path.end(), [](const auto &component){
            return component == "outputs";
        });
    }

    /**
     * Determine whether the supplied path points to a supported image file
     * @param path The path whose file extension will be evaluated
     * @return true when the extension matches one of the supported formats
     */
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

    /**
     * Draw the detection bounding box and facial landmarks on an image
     * @param image The image to annotate in-place
     * @param face The detected face rectangle containing bbox and landmark data
     */
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
}

/**
 * Main function
 * Pls set working directory to project root
 * @return int 0 on success, non-zero on failure
 */
int main(){
    /* =============================== File System =================================== */
    const std::filesystem::path start_dir = std::filesystem::current_path();
    const std::filesystem::path output_dir = std::filesystem::absolute(
            start_dir / "outputs");

    // Resolve image file paths reachable from the working directory
    std::vector<std::filesystem::path> resolved_paths;
    try{
        const auto options = std::filesystem::directory_options::skip_permission_denied;
        for(auto entry = std::filesystem::recursive_directory_iterator(start_dir,
                                                                       options);
            entry != std::filesystem::recursive_directory_iterator(); ++ entry){
            const auto &path = entry->path();

            if(entry->is_directory()){
                if(contains_outputs_component(path)){
                    entry.disable_recursion_pending();
                }
                continue;
            }

            if(! entry->is_regular_file()){
                continue;
            }
            if(contains_outputs_component(path)){
                continue;
            }
            if(is_supported_image_file(path)){
                resolved_paths.push_back(path);
            }
        }
    } catch(const std::filesystem::filesystem_error &err){
        std::cerr << "Failed to enumerate images: " << err.what() << std::endl;
        return 1;
    }

    // Create the `outputs/` directory on-demand without throwing on failure
    std::filesystem::create_directories(output_dir);

    for(const auto &image_path: resolved_paths){
        /* ============================ Image Loading ================================ */
        // Load each image into memory; skip if OpenCV fails to decode it
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if(image.empty()){
            std::cerr << "Failed to read image: " << image_path << std::endl;
            continue;
        }

        std::cout << "Processing image: " << image_path << std::endl;

        /* ============================ Face Detection =============================== */
        // Run CNN face detector over the pixel buffer
        std::vector<FaceRect> faces = objectdetect_cnn(image.data,
                                                       image.cols,
                                                       image.rows,
                                                       static_cast<int>(image.step));

        if(faces.empty()){
            std::cout << "No faces detected." << std::endl;
        } else{
            std::cout << "Detected " << faces.size() << " faces." << std::endl;
        }

        for(size_t idx = 0; idx < faces.size(); ++ idx){
            const FaceRect &face = faces[idx];
            std::cout << "  Face " << idx + 1 << ": score=" << std::fixed
                      << std::setprecision(3)
                      << face.score << " bbox=[" << face.x << ", " << face.y
                      << ", " << face.w << ", " << face.h << "]" << std::endl;
            // Visualize the detection directly on the image buffer
            draw_face_rect(image, face);
        }

        /* ============================ Result Saving ================================ */
        std::filesystem::path output_path =
                output_dir / (image_path.stem().string() + "_result" +
                              image_path.extension().string());
        if(! cv::imwrite(output_path.string(), image)){
            std::cerr << "Failed to save detection result: " << output_path << std::endl;
        } else{
            std::cout << "Result saved to: " << output_path << std::endl;
        }
    }
    return 0;
}
