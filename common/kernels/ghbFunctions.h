#ifndef _GHB_FUNC_H
#define _GHB_FUNC_H
__device__ __forceinline__ void updateGHB(float* mem_arr,float new_val)
{
    mem_arr[0] = mem_arr[1];
    mem_arr[1] = mem_arr[2];
    mem_arr[2] = (new_val);
}

#define NORM_MIN    (-53.0)
#define NORM_MAX    (55.0)
__device__ __forceinline__ float hashGHB(float* mem_arr)
{
    return ((mem_arr[2] - mem_arr[1]) - NORM_MIN) / (NORM_MAX - NORM_MIN);
    float sum = mem_arr[0] + mem_arr[1] + mem_arr[2];
    sum /= 3.0;
    /*
       if (threadIdx.x == 0 && threadIdx.y == 0)
       printf("returning hash of %0.5f\n",sum);
     */
    return sum;
}
#endif
