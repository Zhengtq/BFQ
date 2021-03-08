#include "convolution.h"
#include <iostream>
using namespace std;

namespace ncnn {

Convolution::Convolution() {
    printf("Convolution constructed\n");
    one_blob_only = true;
    support_inplace = false;
}

int Convolution::load_param(const ParamDict& pd) {
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());
    impl_type = pd.get(17, 0);

    return 0;
}

int Convolution::load_model(const ModelBin& mb) {
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty()) return -100;

    /*     float* tmp = (float*)weight_data; */
    // for (int i = 0; i < 10; i++) {
    // cout << tmp[i] << " ";
    // }
    /* cout << endl; */

    if (bias_term) {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty()) return -100;
    }
    return 0;
}

int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
#ifdef FORWARD_LOG
    cout << "conv forward" << endl;
#endif

    //采用了4字节对齐，所以说会有差别
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    /*     if (tops[0] == 62) { */
    // float* tmpptr = top_blob.channel(0);
    // float* tmpptr1 = (float*)bottom_blob.data;
    // //     cout << num_output << "  " << outh << "  " << outw << endl;
    // for (int i = 0; i < 48 * 5 * 5; i++) cout << i << "  " << tmpptr1[i] << endl;
    // cout << w << "  " << h << "  " << channels << endl;
    // cout << pad_top << "  " << pad_bottom << "  " << pad_left << "  " << pad_right << endl;
    // exit(0);
    /* } */
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_borderd;
    make_padding(bottom_blob, bottom_blob_borderd, opt);

    if (bottom_blob_borderd.empty()) return -100;

    w = bottom_blob_borderd.w;
    h = bottom_blob_borderd.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;

    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    for (int p = 0; p < num_output; p++) {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++) {
            for (int j = 0; j < outw; j++) {
                float sum = 0.f;
                if (bias_term) sum = bias_data[p];
                const float* kptr = (const float*)weight_data + maxk * channels * p;

                for (int q = 0; q < channels; q++) {
                    const Mat m = bottom_blob_borderd.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++) {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                        /*                         if (tops[0] == 68) { */
                        // cout << val << endl;
                        /* } */
                    }
                    kptr += maxk;
                }

                outptr[j] = sum;
                /*                 if (tops[0] == 68 && j == 0) { */
                // cout << outh << "  " << outw << endl;
                // cout << "----->" << outptr[0] << endl;
                // exit(0);
                /* } */
            }
            outptr += outw;
        }
    }

/*     if (tops[0] == 68) { */
        // float* tmpptr = (float*)top_blob.data;
        // float* tmpptr1 = (float*)bottom_blob.data;
        // //     cout << num_output << "  " << outh << "  " << outw << endl;
        // int stepcount = 0;
        // int tmpcount = 0;
        // size_t totalsize = alignSize(w * h, 4);
        // for (int i = 0; i < 1000 + 1000; i++) {
            // if (stepcount % 25 == 0 && stepcount != 0) {
                // i += 3;
                // stepcount = 0;
            // }

            // printf("%d  %.3f\n", tmpcount, tmpptr[i]);
            // tmpcount += 1;
            // stepcount++;
            // if (tmpcount == 5 * 5 * 48) break;
        // }
         // cout << w << "  " << h << "  " << channels << endl;
        // //  cout << pad_top << "  " << pad_bottom << "  " << pad_left << "  " << pad_right << endl;
        // cout << "convolution innner out" << totalsize << endl;
        // //    exit(0);
    /* } */

    return 0;
}

void Convolution::make_padding(const Mat& bottom_blob, Mat& bottom_blob_borderd,
                               const Option& opt) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    bottom_blob_borderd = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0) {
        Option opt_b = opt;
        copy_make_border(bottom_blob, bottom_blob_borderd, pad_top, pad_bottom, pad_left,
                         pad_right, BORDER_CONSTANT, pad_value, opt_b);
    } else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233) {
        Option opt_b = opt;
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0) {
            Option opt_b = opt;
            copy_make_border(bottom_blob, bottom_blob_borderd, hpad / 2, hpad - hpad / 2, wpad / 2,
                             wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    } else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234) {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0) {
            Option opt_b = opt;
            copy_make_border(bottom_blob, bottom_blob_borderd, hpad - hpad / 2, hpad / 2,
                             wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}
}
/*                 if (activation_type == 1) { */
// sum = std::max(sum, 0.f);
// } else if (activation_type == 2) {
// float slope = activation_params[0];
// sum = sum > 0.f ? sum : sum * slope;
// } else if (activation_type == 3) {
// float min = activation_params[0];
// float max = activation_params[1];
// if (sum < min) sum = min;
// if (sum > max) sum = max;
// } else if (activation_type == 4) {
// sum = static_cast<float>(1.f / (1.f + exp(-sum)));
// } else if (activation_type == 5) {
// const float MISH_THRESHOLD = 20;
// float x = sum, y;
// if (x > MISH_THRESHOLD)
// y = x;
// else if (x < -MISH_THRESHOLD)
// y - expf(x);
// else
// y = logf(expf(x) + 1);
// sum = static_cast<float>(x * tanh(y));
/* } */
