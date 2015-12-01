Interoperability with CUDA {#interop_cuda}
========

As extensive as ArrayFire is, there are a few cases where you are still working with custom [CUDA] (@ref interop_cuda) or [OpenCL] (@ref interop_opencl) kernels. For example, you may want to integrate ArrayFire into an existing code base for productivity or you may want to keep it around the old implementation for testing purposes. Arrayfire provides a number of functions that allow it to work alongside native CUDA commands. In this tutorial we are going to talk about how to use native CUDA memory operations and integrate custom CUDA kernels into ArrayFire in a seamless fashion.

# In and Out of Arrayfire

First, let's consider the following code and then break it down bit by bit.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
int main() {
    af::array x = randu(num);
    af::array y = randu(num);

    float *d_x = x.device<float>();
    float *d_y = y.device<float>();

    af::sync();

    // Launch kernel to do the following operations
    // y = sin(x)^2 + cos(x)^2
    launch_simple_kernel(d_x, d_y, num);

    x.unlock();
    y.unlock();

    // check for errors, should be 0,
    // since sin(x)^2 + cos(x)^2 == 1
    float err = af::sum(af::abs(y-1));
    printf("Error: %f\n", err);
    return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Breakdown
Most kernels require an input. In this case, we created a random uniform array **x**.
We also go ahead and prepare the output array. The necessary memory required is allocated in array **y** before the kernel launch.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    af::array x = randu(num);
    af::array y = randu(num);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, the output is the same size as in the input. Note that the actual output data type is not specified. For such cases, ArrayFire assumes the data type is single precision floating point ( af::f32 ). If necessary, the data type can be specified at the end of the array(..) constructor. Once you have the input and output arrays, you will need to extract the device pointers / objects using array::device() method in the following manner.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    float *d_x = x.device<float>();
    float *d_y = y.device<float>();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accesing the device pointer in this manner internally sets a flag prohibiting the arrayfire object from further managing the memory. Ownership will need to be returned to the af::array object once we are finished using it.

Before  launching your custom kernel, it is best to make sure that all ArrayFire computations have finished. This can be called by using af::sync(). The function ensures you are not unintentionally doing out of order executions.
af::sync() is not strictly required if you are not using streams in CUDA.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    af::sync();

    // Launch kernel to do the following operations
    // y = sin(x)^2 + cos(x)^2
    launch_simple_kernel(d_x, d_y, num);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The function **launch_simple_kernel** handles the launching of your custom kernel. We will have a look at how to do this in CUDA and OpenCL later in the post.

Once you have finished your computations, you have to tell ArrayFire to take control of the memory objects.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    x.unlock();
    y.unlock();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a very crucial step as ArrayFire believes the user is still in control of the pointer. This means that ArrayFire will not perform garbage collection on these objects resulting in memory leaks. You can now proceed with the rest of the program. In our particular example, we are just performing an error check and exiting.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    // check for errors, should be 0,
    // since sin(x)^2 + cos(x)^2 == 1
    float err = af::sum(af::abs(y-1));
    printf("Error: %f\n", err);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Launching a CUDA kernel
Integrating a CUDA kernel into your ArrayFire code base is a fairly straightforward process. You need to set the launch configuration parameters, launch the kernel and wait for the computations to finish. This is shown below.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 
 __global__
 static void simple_kernel(float *d_y,
                           const float *d_x,
                           const int num)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < num) {
        float x = d_x[id];
        float sin_x = sin(x);
        float cos_x = cos(x);
        d_y[id] = (sin_x * sin_x) + (cos_x * cos_x);
    }
}

void inline launch_simple_kernel(float *d_y,
                                 const float *d_x,
                                 const int num)
{
    // Set launch configuration
    const int threads = 256;
    const int blocks = (num / threads) + ((num % threads) ? 1 : 0);
    simple_kernel<<<blocks, threads>>>(d_y, d_x, num);
    // Synchronize and check for error
    cudaDeviceSynchronize();
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Additional interop functions and CUDA Streams

Arrayfire provides a collection of CUDA interoperability functions for additional capabilities when working with custom CUDA code. To use them, we need to include the appropriate header.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <af/cuda.h>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing these headers allow us to do are to get and set the active device using native CUDA device ids. This is achieved through the following functions:
    **static int getNativeId (int id)** -- Get the native device id of the CUDA device with id in the ArrayFire context.
    **static void setNativeId (int nativeId)**  -- Set the CUDA device with given native id as the active device for ArrayFire. 

These functions are available within the afcu:: namespace and equal C variants can be fund in the full [cuda interop documentation.](group__cuda__mat.htm)