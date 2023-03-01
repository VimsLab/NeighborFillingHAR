#pragma once
#include <torch/extension.h>

at::Tensor k_query(at::Tensor new_xyz, at::Tensor xyz,
                      const int nsample);
