// main.cpp — usage example
#include "object_labeller.hpp"

int main(int argc, char* argv[]) {
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) return 1;

    ObjectLabeller::Config cfg;
    cfg.confThresh = 0.65f;
    cfg.nmsOverlap = 0.40f;

    ObjectLabeller labeller(cfg);
    labeller.addClassifier(std::make_unique<SVMClassifier>("car",        "models/car_svm.xml"));
    labeller.addClassifier(std::make_unique<SVMClassifier>("pedestrian", "models/ped_svm.xml"));
    labeller.addClassifier(std::make_unique<SVMClassifier>("cyclist",    "models/cyc_svm.xml"));

    auto dets = labeller.detect(img);

    for (const auto& d : dets) {
        std::cout << d.label << " [" << std::round(d.confidence * 100) << "%] "
                  << "@ " << d.bbox << "\n";

        cv::Scalar color = (d.label == "pedestrian") ? cv::Scalar(203,83,212)
                         : (d.label == "car")        ? cv::Scalar(221,138,55)
                         :                             cv::Scalar(55,158,221);
        cv::rectangle(img, d.bbox, color, 2);
        cv::putText(img, d.label + " " + std::to_string(int(d.confidence*100)) + "%",
                    d.bbox.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    cv::imwrite("output.jpg", img);
}