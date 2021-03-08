#ifndef LAYER_SPLIT_H
#define LAYER_SPLIT_H

#include "layer.h"

namespace ncnn {

class Split : public Layer {
  public:
    Split();
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs,
                        const Option& opt) const;
};
}

#endif
