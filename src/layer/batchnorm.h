#ifndef LAYER_BATCHNORM_H
#define LAYER_BATCHNORM_H

#include "layer.h"

namespace ncnn {

class BatchNorm : public Layer {
  public:
    BatchNorm();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

  public:
    int channels;
    float eps;
    Mat slope_data;
    Mat mean_data;
    Mat var_data;
    Mat bias_data;

    Mat a_data;
    Mat b_data;
};
}

#endif
