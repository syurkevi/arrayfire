/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#if defined (WITH_GRAPHICS)

#include <af/defines.h>
#include <Array.hpp>
#include <image.hpp>
#include <err_cpu.hpp>
#include <graphics_common.hpp>
#include <platform.hpp>
#include <kernel/moments.hpp>
#include <queue.hpp>

using af::dim4;

namespace cpu
{

template<typename T>
void copy_image(const Array<T> &in, const fg::Image* image)
{
    in.eval();
    getQueue().sync();
    CheckGL("Before CopyArrayToPBO");
    const T *d_X = in.get();
    size_t data_size = image->size();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image->pbo());
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, data_size, d_X);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CheckGL("In CopyArrayToPBO");
}

template<typename T>
void moments(T* val, const Array<T> &in, const af_moment moment)
{
    in.eval();
    getQueue().sync();

    switch(moment) {
        case M00:
            getQueue().enqueue(kernel::moments<T, M00>, val, in);
            break;
        case M01:
            getQueue().enqueue(kernel::moments<T, M01>, val, in);
            break;
        case M10:
            getQueue().enqueue(kernel::moments<T, M10>, val, in);
            break;
        case M11:
            getQueue().enqueue(kernel::moments<T, M11>, val, in);
            break;
        default:  break;
    }

    getQueue().sync();

}


#define INSTANTIATE(T)  \
    template void copy_image<T>(const Array<T> &in, const fg::Image* image);            \
    template void moments<T>(T* val, const Array<T> &in, const af_moment moment);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}

#endif  // WITH_GRAPHICS
