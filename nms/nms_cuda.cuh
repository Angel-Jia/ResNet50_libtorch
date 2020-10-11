#include "torch/torch.h"


torch::Tensor nms_cuda(const torch::Tensor& dets, const float threshold);

torch::Tensor nms(const torch::Tensor& dets, const float threshold){
    return nms_cuda(dets, threshold);
}
