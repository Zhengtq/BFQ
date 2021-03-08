#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "allocator.h"
#include "blob.h"
#include "datareader.h"
#include "iostream"
#include "layer.h"
#include "mat.h"
#include "net.h"
using namespace std;

int main(int argc, char** argv) {
    ncnn::Net mynet;
    mynet.load_param("../examples/filter.param");
    mynet.load_model("../examples/sfnv2.bin");

    cout << "LOAD FINISH" << endl << endl;

    const char* imagepath = "../examples/80x80.bmp";
    cv::Mat bgr = cv::imread(imagepath, 1);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows);

    for (int j = 0; j < 1; j++) {
        ncnn::Extractor ex = mynet.create_extractor();
        ex.input("blob_0", in);
        ncnn::Mat out;
        ex.extract("blob_194", out);

        float* tmp = (float*)out;
        float all_sum = 0;
        for (int i = 0; i < 1; i++) {
            cout << i << "  " << tmp[i] << endl;
        }
    }
    printf("yes\n");
    return 0;
}
