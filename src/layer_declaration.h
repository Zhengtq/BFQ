#ifndef NCNN_LAYER_DECLARATION
#define NCNN_LAYER_DECLARATION
#include "layer/relu.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(ReLU)
}
#include "layer/input.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Input)
}

#include "layer/pooling.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Pooling)
}

#include "layer/convolution.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Convolution)
}

#include "layer/split.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Split)
}

#include "layer/concat.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Concat)
}

#include "layer/dropout.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Dropout)
}

#include "layer/softmax.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Softmax)
}

#include "layer/padding.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(Padding)
}

#include "layer/batchnorm.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(BatchNorm)
}

#include "layer/convolutiondepthwise.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(ConvolutionDepthWise)
}

#include "layer/shuffletwo.h"
namespace ncnn {
DEFINE_LAYER_CREATOR(ShuffleTwo)
}

#endif
