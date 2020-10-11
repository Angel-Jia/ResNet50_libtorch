#include "torch/torch.h"

at::Tensor ROIAlignForwardV2Laucher(const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio, bool aligned);

inline at::Tensor ROIAlign_forwardV2(const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio, bool aligned) {
return ROIAlignForwardV2Laucher(input, rois, spatial_scale, pooled_height,
   pooled_width, sampling_ratio, aligned);
}
