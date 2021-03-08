#ifndef LAYER_SHUFFLETWO_H
#define LAYER_SHUFFLETWO_H

#include "layer.h"

namespace ncnn {

class ShuffleTwo : public Layer {
  public:
    ShuffleTwo();
    virtual int forward(const const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs,
                        const Option& opt) const;
};
}

#endif
