/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cstdio>
#include <math.h>
#include <iostream>

using namespace af;
using namespace std;

static const int ITERATIONS = 5;
static const float PRECISION = 1.0f/ITERATIONS;

int main(int argc, char *argv[])
{
    try {
        // Initialize the kernel array just once
        af::info();
        af::Window myWindow(512, 512, "3D Surface Example: ArrayFire");

        array X = seq(-af::Pi, af::Pi, PRECISION);
        array Y = transpose(seq(-af::Pi, af::Pi, PRECISION));
        unsigned xs = X.dims(0);
        unsigned ys = Y.dims(1);
        cout<<xs<<ys<<endl;
        array Z =randn(xs,ys);// tile(seq(1,X.dims(0)), 1, Y.dims(0));

        X = tile(X, 1, ys);
        Y = tile(Y, xs);
        af_print(X);
        af_print(Y);
        //Y = flat(transpose(tile(Y, 1, ys)));

        static float t=0;
        do{
            Z = sin(X) + cos(Y);
            af_print(Z);
            //Z = moddims(Z, af::dim4(xs, ys));
            //myWindow.surface(Z, NULL);
            myWindow.surface(X, Y, Z, NULL);
            t+=0.1;
        } while (!myWindow.close());
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
    #endif
    return 0;
}

