#include "mat.h"

#include <math.h>
#include "layer.h"

namespace ncnn {

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type,
                      float v, const Option& opt) {
    Layer* padding = create_layer("Padding");
    ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);
    padding->forward(src, dst, opt);
    delete padding;
}
}
