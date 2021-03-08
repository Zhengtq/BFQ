#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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
namespace py = pybind11;

cv::Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char>& input) {
    if (input.ndim() != 2) throw std::runtime_error("1-channel image must be 2 dims ");
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    return mat;
}

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {
    if (input.ndim() != 3) throw std::runtime_error("3-channel image must be 3 dims ");
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    return mat;
}

class BFQ {
  public:
    BFQ();
    ~BFQ();
    int Run(py::array_t<unsigned char>& input);
    int Uninit();
    ncnn::Net bfqnet;
    float final_result;
    float get_result();
    string get_version();
};

BFQ::BFQ() {
    bfqnet.load_param("./filter.param");
    bfqnet.load_model("./sfnv2.bin");
    final_result = 0.0;
}

BFQ::~BFQ() {}



int BFQ::Run(py::array_t<unsigned char>& input) {
    cv::Mat src_img;
    if (input.ndim() == 2)
        src_img = numpy_uint8_1c_to_cv_mat(input);
    else if (input.ndim() == 3)
        src_img = numpy_uint8_3c_to_cv_mat(input);
    else {
        fprintf(stderr, "This image is not valid\n");
        return -2;
    }

    if (src_img.cols <= 0 || src_img.rows <= 0) {
        fprintf(stderr, "This image is not valid\n");
        return -2;
    }
    ncnn::Mat in =
        ncnn::Mat::from_pixels(src_img.data, ncnn::Mat::PIXEL_BGR, src_img.cols, src_img.rows);

    ncnn::Extractor ex = bfqnet.create_extractor();
    ex.input("blob_0", in);
    ncnn::Mat out;
    ex.extract("blob_194", out);

    float* tmp = (float*)out;
    final_result = tmp[0];
    return 0;
}

float BFQ::get_result() { return final_result; }

string BFQ::get_version() {
    string version = "v1.2.0.1";
    return version;
}
PYBIND11_MODULE(bfq, m) {
    pybind11::class_<BFQ>(m, "BFQ")
        .def(pybind11::init())
        .def("Run", &BFQ::Run)
        .def("get_result", &BFQ::get_result)
        .def("get_version", &BFQ::get_version);
}
