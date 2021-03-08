#ifndef LAYER_PADDING_H
#define LAYER_PADDING_H
#include "layer.h"

namespace ncnn {

class Padding : public Layer {
  public:
    Padding();
    virtual int load_param(const ParamDict& pd);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
/*     virtual int forward(const std::vector<Mat>& bottom_blob, std::vector<Mat>& top_blob, */
                        /* const Option& opt) const; */

  public:
    int top;
    int bottom;
    int left;
    int right;
    int type;
    float value;
    int front;
    int behind;

    int per_channel_pad_data_size;
    Mat per_channel_pad_data;
};

}


#endif

