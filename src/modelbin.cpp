#include "modelbin.h"
#include <string.h>
#include <iostream>
#include "datareader.h"
using namespace std;

namespace ncnn {

ModelBin::~ModelBin() {}

ModelBinFromDataReader::ModelBinFromDataReader(const DataReader& _dr) : dr(_dr) {}

Mat ModelBinFromDataReader::load(int w, int type) const {
    //    cout << w << "  " << type << endl;
    if (type == 0) {
        size_t nread;

        /*         union { */
        // struct {
        // unsigned char f0;
        // unsigned char f1;
        // unsigned char f2;
        // unsigned char f3;
        // };
        // unsigned int tag;

        // } flag_struct;

        // nread = dr.read(&flag_struct, sizeof(flag_struct));
        /* unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;
         */
        int flag = 0;
        Mat m(w);
        if (m.empty()) return m;
        if (flag != 0) {
            return m;
        } else if (1) {
            nread = dr.read(m, w * sizeof(float));

   /*          float* tmp = (float*)m; */
            // for (int i = 0; i < 100; i++) {
                // cout << i << "  " << tmp[i] << endl;
            // }
            /* exit(0); */
        }
        return m;
    } else if (type == 1) {
        Mat m(w);
        if (m.empty()) return m;
        size_t nread = dr.read(m, w * sizeof(float));
        return m;
    }
    return Mat();
}
}
