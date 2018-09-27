
#include <iostream>
#include <string>
#include <cassert>
#include <opencv2/dnn.hpp>

void printBlob(cv::Mat blob) {
    assert(blob.dims == 4);
    assert(blob.size[0] == 1); // n == 1
    assert(blob.size[1] == 1); // c == 1
    for(int h = 0; h < blob.size[2]; ++h) {
        for(int w = 0; w < blob.size[3]; ++w) {
            auto pos = cv::Vec<int, 4>(0, 0, h, w);
            std::cout << blob.at<float>(pos) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void testNet(std::string modelFile) {
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelFile);

    cv::Mat zeros = cv::Mat::zeros(cv::Size(5, 5), CV_32F);
    cv::Mat input = cv::dnn::blobFromImage(zeros);
    net.setInput(input);
    cv::Mat output = net.forward();

    std::cout << modelFile << std::endl;
    printBlob(input);
    printBlob(output);
}


int main() {
    testNet("valid.h5.pb");
    testNet("same.h5.pb");
    return 0;
}