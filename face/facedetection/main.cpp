#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    CascadeClassifier face_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        cout << "Error loading face cascade!" << endl;
        return -1;
    }

    VideoCapture cap(0);
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    if (!cap.isOpened()) {
        cout << "Cannot open USB webcam!" << endl;
        
        for (int i = 1; i <= 3; i++) {
            cap.open(i);
            if (cap.isOpened()) {
                cout << "Found camera at index " << i << endl;
                break;
            }
        }
        
        if (!cap.isOpened()) {
            cout << "Failed to open camera" << endl;
            return -1;
        }
    }

    cout << "Connected to webcam!" << endl;

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Failed to grab frame" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);
        
        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(255, 0, 0), 2);
        }
        
        imshow("USB Webcam - Face Detection", frame);
        if (waitKey(1) == 'q') break;
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}



