#include "concat.h"
#include <iostream>
using namespace std;

namespace ncnn {

Concat::Concat() {
    printf("Concat constructed\n");
    one_blob_only = false;
    support_inplace = false;
}

int Concat::load_param(const ParamDict& pd) {
    axis = pd.get(0, 0);
    return 0;
}

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs,
                    const Option& opt) const {
#ifdef FORWARD_LOG
    cout << "concat forward" << endl;
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++) {
        top_blobs[i] = bottom_blob;
    }

    int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;

    if (dims == 1) {
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            top_w += bottom_blobs[b].w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        unsigned char* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat& bottom_blob = bottom_blobs[b];

            int w = bottom_blob.w;
            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, w * elemsize);
            outptr += w * elemsize;
        }
        return 0;
    }

    if (dims == 2 && axis == 0) {
        int w = bottom_blobs[0].w;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            top_h += bottom_blobs[b].h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        unsigned char* output = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat& bottom_blob = bottom_blobs[b];
            int size = w * bottom_blob.h;
            const unsigned char* ptr = bottom_blob;
            memcpy(output, ptr, size * elemsize);
            output += size + elemsize;
        }

        return 0;
    }

    if (dims == 2 && axis == 1) {
        int h = bottom_blobs[0].h;
        int top_w = 0;

        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            top_w += bottom_blobs[b].w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        for (int i = 0; i < h; i++) {
            unsigned char* outptr = top_blob.row<unsigned char>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++) {
                const Mat& bottom_blob = bottom_blobs[b];
                const unsigned char* ptr = bottom_blob.row<const unsigned char>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);
                outptr += bottom_blob.w * elemsize;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 0) {
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            top_channels += bottom_blobs[b].c;
        }


        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        int q = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            const Mat& bottom_blob = bottom_blobs[b];
            int channels = bottom_blob.c;
            size_t size = bottom_blob.cstep * channels;
            const unsigned char* ptr = bottom_blob;
            unsigned char* outptr = top_blob.channel(q);
            memcpy(outptr, ptr, size * elemsize);
            q += channels;
        }

        return 0;
    }

    if (dims == 3 && axis == 1) {
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;

        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            top_h += bottom_blobs[b].h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        for (int q = 0; q < channels; q++) {
            unsigned char* outptr = top_blob.channel(q);
            for (size_t b = 0; b < bottom_blobs.size(); b++) {
                const Mat& bottom_blob = bottom_blobs[b];
                int size = bottom_blob.w * bottom_blob.h;
                const unsigned char* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);
                outptr += size * elemsize;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 2) {
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        int top_w = 0;

        for (size_t b = 0; b < bottom_blobs.size(); b++) {
            top_w = bottom_blobs[b].w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        for (int q = 0; q < channels; q++) {
            unsigned char* outptr = top_blob.channel(q);
            for (int i = 0; i < h; i++) {
                for (size_t b = 0; b < bottom_blobs.size(); b++) {
                    const Mat& bottom_blob = bottom_blobs[b];
                    const unsigned char* ptr = bottom_blob.channel(q).row<const unsigned char>(i);
                    memcpy(outptr, ptr, bottom_blob.w * elemsize);
                    outptr += bottom_blob.w * elemsize;
                }
            }
        }
        return 0;
    }
    
    return 0;
}
}
