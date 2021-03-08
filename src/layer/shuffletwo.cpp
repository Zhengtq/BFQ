#include "shuffletwo.h"

namespace ncnn {

ShuffleTwo::ShuffleTwo() {
    printf("ShuffleTwo constructed\n");
    one_blob_only = false;
    support_inplace = false;
}

int ShuffleTwo::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs,
                        const Option& opt) const {
#ifdef FORWARD_LOG
    printf("shuffletwo forward\n");
#endif

    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;
    int top_channels = 0;
    size_t elemsize = bottom_blobs[0].elemsize;
    int channel_size = h * w;
    size_t transfer_size = alignSize(channel_size, 4);
    //  size_t transfer_size = channel_size;

    for (size_t b = 0; b < bottom_blobs.size(); b++) {
        const Mat& bottom_blob = bottom_blobs[b];
        top_channels += bottom_blob.c;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    int now_shuffle_c = 0;
    int now_single_c = 0;
    while (now_shuffle_c < top_channels) {
        for (int b = 0; b < bottom_blobs.size(); b++) {
            unsigned char* outptr = top_blob.channel(now_shuffle_c);
            const Mat& bottom_blob = bottom_blobs[b];
            const unsigned char* ptr = bottom_blob.channel(now_single_c);
            memcpy(outptr, ptr, transfer_size * elemsize);
            now_shuffle_c += 1;
        }
        now_single_c += 1;
    }

    // if (tops[0] == 65) {
        // float* tmpptr = (float*)bottom_blobs[0].data;
        // float* tmpptr1 = (float*)bottom_blobs[1].data;
        // //     cout << num_output << "  " << outh << "  " << outw << endl;
        // int tmpcount = 0;
        // for (int i = 0; i < 5 * 5 * 48 + 240; i++) {
                // if (tmpptr[i] == 0) continue;
            // printf("%d  %.3f\n", tmpcount, tmpptr[i]);
            // //   if (tmpcount == 676) break;
            // tmpcount += 1;
        // }
        // exit(0);
    /* } */

    return 0;
}
}
