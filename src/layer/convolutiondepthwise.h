#ifndef LAYER_CONVOLUTIONDEPTHWISE_H
#define LAYER_CONVOLUTIONDEPTHWISE_H

#include <math.h>
#include "layer.h"

namespace ncnn {

class ConvolutionDepthWise : public Layer {
  public:
    ConvolutionDepthWise();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

  protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;

  public:
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_h;
    int stride_w;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;
    int bias_term;
    int weight_data_size;
    int group;
    int activation_type;
    Mat activation_params;
    Mat weight_data;
    Mat bias_data;
};
}

#endif
