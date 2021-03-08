#include "relu.h"
#include <iostream>
using namespace std;

namespace ncnn {

ReLU::ReLU() {
    printf("relu CONSTRUCTED\n");
    one_blob_only = true;
    support_inplace = true;
}

int ReLU::load_param(const ParamDict& pd) {
    slope = pd.get(0, 0.f);
    return 0;
}

int ReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
#ifdef FORWARD_LOG
    cout << "relu forward" << endl;
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (slope == 0.f) {
        for (int q = 0; q < channels; q++) {
            float* ptr = bottom_top_blob.channel(q);
            for (int i = 0; i < size; i++) {
                if (ptr[i] < 0) ptr[i] = 0;
            }
        }

    } else {
        for (int q = 0; q < channels; q++) {
            float* ptr = bottom_top_blob.channel(q);
            for (int i = 0; i < size; i++) {
                if (ptr[i] < 0) ptr[i] *= slope;
            }
        }
    }
    return 0;
}
}
