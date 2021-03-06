#include "pooling.h"
#include <math.h>
#include <iostream>
#define FLT_MAX 3.402823466e+38F
using namespace std;

namespace ncnn {

Pooling::Pooling() {
    printf("Pooling CONSTRUCTED\n");
    one_blob_only = true;
    support_inplace = false;
}

int Pooling::load_param(const ParamDict& pd) {
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);
    avgpool_count_include_pad = pd.get(6, 0);
    return 0;
}

int Pooling::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
#ifdef FORWARD_LOG
    cout << "pooling forward" << endl;
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (global_pooling) {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX) {
            for (int q = 0; q < channels; q++) {
                const float* ptr = bottom_blob.channel(q);
                float max = ptr[0];
                for (int i = 0; i < size; i++) {
                    max = std::max(max, ptr[i]);
                }
                top_blob[q] = max;
            }
        } else if (pooling_type == PoolMethod_AVE) {
            for (int q = 0; q < channels; q++) {
                const float* ptr = bottom_blob.channel(q);
                float sum = 0.f;
                for (int i = 0; i < size; i++) {
                    sum += ptr[i];
                }
                top_blob[q] = sum / size;
            }
        }

 /*        int all_size = channels; */
        // for (int i = 0; i < all_size; i++) {
            // cout << i << "  " << top_blob[i] << endl;
        /* } */



        return 0;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty()) return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    const int maxk = kernel_w * kernel_h;
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX) {
        for (int q = 0; q < channels; q++) {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    const float* sptr = m.row(i * stride_h) + j * stride_w;
                    float max = sptr[0];
                    for (int k = 0; k < maxk; k++) {
                        float val = sptr[space_ofs[k]];
                        max = std::max(max, val);
                    }
                    outptr[j] = max;
                }
                outptr += outw;
            }
        }
    } else if (pooling_type == PoolMethod_AVE) {
        if (avgpool_count_include_pad == 0) {
            int wtailpad = 0;
            int htailpad = 0;
            if (pad_mode == 0) {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
            }

            for (int q = 0; q < channels; q++) {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++) {
                    int sy0 = i * stride_h;
                    for (int j = 0; j < outw; j++) {
                        int sx0 = j * stride_w;
                        float sum = 0;
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++) {
                            int sy = sy0 + ki;
                            if (sy < pad_top) continue;
                            if (sy >= h - pad_bottom - htailpad) break;

                            for (int kj = 0; kj < kernel_w; kj++) {
                                int sx = sx0 + kj;
                                if (sx < pad_left) continue;
                                if (sx > w - pad_left - wtailpad) break;

                                float val = m.row(sy)[sx];
                                sum += val;
                                area += 1;
                            }
                        }
                        outptr[j] = sum / area;
                    }
                    outptr += outw;
                }
            }
        } else {
            for (int q = 0; q < channels; q++) {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++) {
                    for (int j = 0; j < outw; j++) {
                        const float* sptr = m.row(i * stride_h) + j * stride_w;
                        float sum = 0;

                        for (int k = 0; k < maxk; k++) {
                            float val = sptr[space_ofs[k]];
                            sum += val;
                        }
                        outptr[j] = sum / maxk;
                    }
                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

void Pooling::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered,
                           const Option& opt) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX) {
        pad_value = bottom_blob.elemsize == 1 ? -128.f : -FLT_MAX;
    } else if (pooling_type == PoolMethod_AVE) {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0) wtailpad = stride_w - wtail;
        if (htail != 0) htailpad = stride_h - htail;
        Option opt_b = opt;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad,
                         pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt_b);
    } else if (pad_mode == 1) {
        Option opt_b = opt;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left,
                         pad_right, BORDER_CONSTANT, pad_value, opt_b);
    } else if (pad_mode == 2) {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0) {
            Option opt_b = opt;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2,
                             wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    } else if (pad_mode == 3) {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0) {
            Option opt_b = opt;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2,
                             wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}
}
