#include "batchnorm.h"
#include <math.h>
#include <iostream>
using namespace std;

namespace ncnn {

BatchNorm::BatchNorm() {
    printf("BatchNorm constructed\n");
    one_blob_only = true;
    support_inplace = true;
}

int BatchNorm::load_param(const ParamDict& pd) {
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.f);
    //  cout << channels << "  " << eps << endl;
    return 0;
}

int BatchNorm::load_model(const ModelBin& mb) {
    slope_data = mb.load(channels, 1);
    if (slope_data.empty()) return -100;
    bias_data = mb.load(channels, 1);
    if (bias_data.empty()) return -100;
    mean_data = mb.load(channels, 1);
    if (mean_data.empty()) return -100;
    var_data = mb.load(channels, 1);
    if (var_data.empty()) return -100;


    a_data.create(channels);
    if (a_data.empty()) return -100;
    b_data.create(channels);
    if (b_data.empty()) return -100;

    for (int i = 0; i < channels; i++) {
        float sqrt_var = static_cast<float>(sqrt(var_data[i] + eps));
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        b_data[i] = slope_data[i] / sqrt_var;
    }
    return 0;
}

int BatchNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
#ifdef FORWARD_LOG
    cout << "batchnorm forward" << endl;
#endif
/*     if (bottoms[0] == 62) { */
        // float* tmpptr1 = (float*)bottom_top_blob.data;
        // int tmpcount = 0;
        // for (int i = 0; i < 1000 + 1000; i++) {
               // if (tmpptr1[i] == 0) continue;
            // printf("%d  %.3f\n", tmpcount, tmpptr1[i]);
            // if (tmpcount == 1199) break;
            // tmpcount += 1;
        // }
        // cout << "batchnorm cout" << endl;
        // exit(0);
    /* } */
    int dims = bottom_top_blob.dims;

    if (dims == 1) {
        int w = bottom_top_blob.w;
        float* ptr = bottom_top_blob;

        for (int i = 0; i < w; i++) {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
    }

    if (dims == 2) {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        for (int i = 0; i < h; i++) {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];
            for (int j = 0; j < w; j++) {
                ptr[j] = b * ptr[j] + a;
            }
        }
    }

    if (dims == 3) {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int size = w * h;

        for (int q = 0; q < channels; q++) {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];
            for (int i = 0; i < size; i++) {
                ptr[i] = b * ptr[i] + a;
            }
        }
    }



    return 0;
}
}
