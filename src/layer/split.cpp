#include "split.h"

namespace ncnn {

Split::Split() {
    printf("Split constructed\n");
    one_blob_only = false;
    support_inplace = false;
}

int Split::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs,
                   const Option& opt) const {
#ifdef FOREARD_LOG
    cout << "split forward" << endl;
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    int bottom_channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int split_num = top_blobs.size();
    int single_top_channel = bottom_channels / split_num;

    size_t totalsize = alignSize(w * h, 4);
    size_t single_top_size = totalsize * single_top_channel;
    for (size_t i = 0; i < top_blobs.size(); i++) {
        Mat& top_blob = top_blobs[i];
        top_blob.create(w, h, single_top_channel, elemsize, opt.blob_allocator);
        unsigned char* outptr = top_blob;

        const unsigned char* ptr = bottom_blob.channel(i * single_top_channel);
        memcpy(outptr, ptr, single_top_size * elemsize);
    }

    return 0;
}
}
