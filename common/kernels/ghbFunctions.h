#ifndef _GHB_FUNC_H
#define _GHB_FUNC_H
__device__ __forceinline__ void updateGHB(int* mem_arr,int new_val)
{
    mem_arr[0] = mem_arr[1];
    mem_arr[1] = mem_arr[2];
    mem_arr[2] = (int)(new_val);
}

__device__ __forceinline__ int hashGHB(int* mem_arr)
{
#if 1
    // delta approximation
#define NORM_MIN    (-53)
#define NORM_MAX    (55)
    return ((mem_arr[2] - mem_arr[1]) - NORM_MIN);
#else
    // delta approximation
#define NORM_MIN    (-53.0)
#define NORM_MAX    (55.0)
    return ((mem_arr[2] - mem_arr[1]) - NORM_MIN) / (NORM_MAX - NORM_MIN);
    // value approximation
#define NORM_MIN    (9.0)
#define NORM_MAX    (255.0)
    return ((mem_arr[2]) - NORM_MIN) / (NORM_MAX - NORM_MIN);
    // old
    float sum = mem_arr[0] + mem_arr[1] + mem_arr[2];
    sum /= 3.0;
    /*
       if (threadIdx.x == 0 && threadIdx.y == 0)
       printf("returning hash of %0.5f\n",sum);
     */
    return sum;
#endif
}
#endif
