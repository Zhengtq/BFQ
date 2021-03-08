#ifndef NCNN_MAT_H
#define NCNN_MAT_H
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "allocator.h"
#include "option.h"
using namespace std;

namespace ncnn {

class Mat {
  public:
    Mat();
    ~Mat();
    Mat(const Mat& m);
    Mat(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    bool empty() const;
    size_t total() const;

    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);

    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);

    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocate = 0);
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);

    void fill(float v);
    void fill(int v);
    template <typename T>
    void fill(T v);

    void release();

    Mat channel(int c);
    const Mat channel(int c) const;

    Mat clone(Allocator* allocator = 0) const;

    //定义重载类型转化符
    template <typename T>
    operator T*();
    template <typename T>
    operator const T*() const;

    float& operator[](size_t i);
    const float& operator[](size_t i) const;

    enum PixelType {
        PIXEL_RGB = 1,
        PIXEL_BGR = 2,
    };

    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h,
                           Allocator* allocator = 0);
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, int stride,
                           Allocator* allocator = 0);

    float* row(int y);
    const float* row(int y) const;
    template <typename T>
    T* row(int y);
    template <typename T>
    const T* row(int y) const;
    Mat& operator=(const Mat& m);

  public:
    void* data;
    int* refcount;
    size_t elemsize;
    int elempack;
    int dims;
    int w;
    int h;
    int c;
    size_t cstep;
    Allocator* allocator;
};

enum BorderType {
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
};

inline Mat::Mat()
    : data(0),
      refcount(0),
      elemsize(0),
      elempack(0),
      allocator(0),
      dims(0),
      w(0),
      h(0),
      c(0),
      cstep(0) {}

inline Mat::Mat(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), dims(0), w(_w), h(0), c(0), cstep(0) {
    create(_w, _elemsize, _allocator);
}

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type,
                      float v, const Option& opt = Option());

inline Mat::Mat(const Mat& m)
    : data(m.data),
      refcount(m.refcount),
      elemsize(m.elemsize),
      elempack(m.elempack),
      allocator(m.allocator),
      dims(m.dims),
      w(m.w),
      h(m.h),
      c(m.c),
      cstep(m.cstep) {
    //此时不用把左值release清空，因为左值已经被赋值
    if (refcount) {
        //    cout << "++++++++++++++++++++++++++++++++2" << endl;
        NCNN_XADD(refcount, 1);
    }
}

inline Mat& Mat::operator=(const Mat& m) {
    if (this == &m) return *this;

    if (m.refcount) {
        NCNN_XADD(m.refcount, 1);
        //     cout << "+++++++++++++++++++++++++1" << endl;
    }
    //   cout << "#### = Release" << endl;

    //这个release的目的是先把左值清空
    release();
    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;
    return *this;
}





inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _elempack,
                Allocator* _allocator)
    : data(_data),
      refcount(0),
      elemsize(_elemsize),
      elempack(_elempack),
      allocator(_allocator),
      dims(2),
      w(_w),
      h(_h),
      c(1) {
    cstep = (size_t)w * h;
}

inline Mat::~Mat() {
    //  cout << "#### ~Mat Release" << endl;
    release(); 
}

inline void Mat::create(int _w, size_t _elemsize, Allocator* _allocator) {
    //如果是初始化Mat那么这个判断不会起到作用，肯定会到下面的函数体的
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    //    cout << "#### create_release" << endl;
    release();
    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;


    dims = 1;
    w = _w;
    h = 1;
    c = 1;
    cstep = w;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        //把refcount的地址放到了data的最后一位，同时赋值为1
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator) {
    //如果是初始化Mat那么这个判断不会起到作用，肯定会到下面的函数体的
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack &&
        allocator == _allocator)
        return;

    release();
    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;
    cstep = w;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        //把refcount的地址放到了data的最后一位，同时赋值为1
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, size_t _elemsize, Allocator* _allocator) {
    //如果是初始化Mat那么这个判断不会起到作用，肯定会到下面的函数体的
    if (dims == 3 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 &&
        allocator == _allocator)
        return;

    release();
    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;
    cstep = (size_t)w * h;
    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        //把refcount的地址放到了data的最后一位，同时赋值为1
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator) {
    //如果是初始化Mat那么这个判断不会起到作用，肯定会到下面的函数体的
    if (dims == 3 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack &&
        elempack == 1 && allocator == _allocator)
        return;

    release();
    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;
    cstep = (size_t)w * h;
    if (total() > 0) {
        // alignSize是尺寸上的对齐，fastMalloc是首地址方面的对齐
        size_t totalsize = alignSize(total() * elemsize, 4);
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        //把refcount的地址放到了data的最后一位，同时赋值为1
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator) {
    //如果是初始化Mat那么这个判断不会起到作用，肯定会到下面的函数体的
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 &&
        allocator == _allocator)
        return;

    release();
    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack,
                        Allocator* _allocator) {
    //如果是初始化Mat那么这个判断不会起到作用，肯定会到下面的函数体的
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize &&
        elempack == _elempack && allocator == _allocator)
        return;

    release();
    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline Mat Mat::clone(Allocator* allocator) const {
    if (empty()) return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize, elempack, allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, allocator);

    if (total() > 0) memcpy(m.data, data, total() * elemsize);

    return m;
}

inline void Mat::fill(float _v) {
    int size = (int)total();
    float* ptr = (float*)data;
    int remain = size;

    for (; remain > 0; remain--) {
        *ptr++ = _v;
    }
}

inline void Mat::fill(int _v) {
    int size = (int)total();
    int* ptr = (int*)data;
    int remain = size;

    for (; remain > 0; remain--) {
        *ptr++ = _v;
    }
}

template <typename T>
inline void Mat::fill(T _v) {
    int size = total();
    T* ptr = (T*)data;
    for (int i = 0; i < size; i++) {
        ptr[i] = _v;
    }
}

inline bool Mat::empty() const { return data == 0 || total() == 0; }
inline size_t Mat::total() const { return cstep * c; }

inline void Mat::release() {


/*     cout << "refcount:" << refcount << "  "; */
    // if (refcount)
        // cout << "*refcount:" << *refcount << endl;
    // else
        /* cout << "*refcount:no" << endl; */
    //当refcount有值，同时又*refcount>1的时候，那么也吧refcount置零
    //这个时候通常是拷贝构造函数或者是赋值构造函数执行的时候
    if (refcount && NCNN_XADD(refcount, -1) == 1) {
        //     NCNN_XADD(refcount, -1);
        //    free(refcount);
        fastFree(data);
    }

    data = 0;
    elemsize = 0;
    elempack = 0;
    dims = 0;
    w = 0;
    h = 0;
    c = 0;
    cstep = 0;
    refcount = 0;
}

inline Mat Mat::channel(int _c) {
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Mat Mat::channel(int _c) const {
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline float* Mat::row(int y) { return (float*)((unsigned char*)data + (size_t)w * y * elemsize); }

inline const float* Mat::row(int y) const {
    return (const float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template <typename T>
inline T* Mat::row(int y) {
    return (T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template <typename T>
inline const T* Mat::row(int y) const {
    return (const T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template <typename T>
inline Mat::operator T*() {
    return (T*)data;
}

template <typename T>
inline Mat::operator const T*() const {
    return (const T*)data;
}

inline float& Mat::operator[](size_t i) { return ((float*)data)[i]; }

inline const float& Mat::operator[](size_t i) const { return ((const float*)data)[i]; }
}
#endif
