#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_k_point_kernel(int b, int n, int m,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  // float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    float diss[1024];
    float indxes[1024];
    // float indxes2[1024];
    int  cnt = 0;
    for (int k = 0; k < n; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = sqrtf((new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z));
      // if (d2 < radius2) {
      diss[k] = d2;
      if (cnt == 0) {
        for (int l = 0; l < nsample; ++l) {
          idx[j * nsample + l] = k;
        }
      }
      // indxes2[k] = k;
      // idx[j * nsample + cnt] = k;
      // ++cnt;
      // }
    }

    for (int s=0;s<n;++s) {
      // out[j*n+s] = dist[j*n+s];
      indxes[s] = s;
    }
    for (int s=0;s<nsample;++s) {
      int min=s;
      // find the min
      for (int t=s+1;t<n;++t) {
          if (diss[t]<diss[min]) {
              min = t;
          }
      }
      // swap min-th and i-th element
      if (min!=s) {
          float tmp = diss[min];
          diss[min] = diss[s];
          diss[s] = tmp;
          // printf("%f\n",tmp);
          int tmpi = indxes[min];
          indxes[min] = indxes[s];
          indxes[s] = tmpi;
      }
    }
    cnt = 0;
    for (int cpy=0;cpy<nsample;cpy++){
      idx[j*nsample+cnt] = indxes[cpy];
      // printf("%d %f\n",idx[j*nsample+cnt], diss[cpy]);
      cnt+=1;
    }
    // printf("\n");
  }
  // printf("\n");
}

void query_k_point_kernel_wrapper(int b, int n, int m,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_k_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}
