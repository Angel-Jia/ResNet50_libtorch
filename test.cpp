#include "torch/script.h"
#include "torch/torch.h"
#include<vector>
#include<map>
#include<string>
#include<cmath>
#include<cassert>
#include <chrono>

#include "nms_cuda.cuh"
#include "roi_align.cuh"
#include <nlohmann/json.hpp>

#include "opencv2/opencv.hpp"

//get files in directory
#include <dirent.h>


using json = nlohmann::json;
using namespace torch::indexing;
using std::vector;
using std::map;
using std::string;
using torch::Tensor;

typedef std::vector<Tensor> TensorVector;
typedef std::tuple<float, float, float, float, float> COORD;


struct ImgMeta {
    std::tuple<size_t, size_t, size_t> img_shape;
    Tensor scale_factor;
    string file_name;
};

vector<std::tuple<string, string>> listdir(string dir_name) {
    vector<std::tuple<string, string>> file_path_infos;
    auto dir = opendir(dir_name.data());
    struct dirent* ent;
    if (dir) {
        while ((ent = readdir(dir)) != NULL) {
            if (0 == strcmp(ent->d_name, "..") || 0 == strcmp(ent->d_name, ".")) {
                continue;
            }
            auto path = std::string(dir_name).append("/").append(ent->d_name);
            string file_name(ent->d_name);
            file_path_infos.push_back(std::make_tuple(path, file_name));
        }
        closedir(dir);
    }
    return file_path_infos;
}

std::tuple<Tensor, Tensor> batched_nms(Tensor bboxes, Tensor scores, Tensor inds,
    bool class_agnostic, float iou_thr) {
    /*
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        bboxes (torch.Tensor): bboxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        inds (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different inds,
            shape (N, ).
        nms_cfg (dict): specify nms type and class_agnostic as well as other
            parameters like iou_thr.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all bboxes,
            regardless of the predicted class

    Returns:
        tuple: kept bboxes and indice.
    """
    */
    Tensor bboxes_for_nms;
    if (class_agnostic) {
        bboxes_for_nms = bboxes;
    }
    else {
        auto max_coordinate = bboxes.max();
        auto offsets = inds.to(bboxes) * (max_coordinate + 1);
        bboxes_for_nms = bboxes + offsets.index({ Slice(), None });
    }

    auto dets_th = torch::cat({ bboxes_for_nms, scores.index({Slice(), None}) }, -1);

    if (dets_th.size(0) == 0) {
        inds = dets_th.new_zeros(0, torch::kLong);
    }
    else {
        inds = nms(dets_th, iou_thr);
    }
    //dets_th 里的bbox和bboxes不一样
    bboxes = bboxes.index(inds);
    scores = dets_th.index({ inds, -1 });
    return std::make_tuple(torch::cat({ bboxes, scores.index({Slice(), None}) }, -1), inds);
}

Tensor delta2bbox(Tensor rois, Tensor deltas, std::tuple<size_t, size_t, size_t> img_shape,
    Tensor means, Tensor stds) {
    /*boxes(torch.Tensor) : Basic boxes.
      pred_bboxes(torch.Tensor) : Encoded boxes with shape
      max_shape(tuple[int], optional) : Maximum shape of boxes.
      Defaults to None.
      wh_ratio_clip(float, optional) : The allowed ratio between
      widthand height.
    */
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rois.device());
    float wh_ratio_clip = 16.0 / 1000.0;

    means = means.repeat({ 1, deltas.size(1) / 4 });
    stds = stds.repeat({ 1, deltas.size(1) / 4 });
    auto denorm_deltas = deltas * stds + means;

    auto dx = denorm_deltas.index({ Slice(), Slice(0, None, 4) });
    auto dy = denorm_deltas.index({ Slice(), Slice(1, None, 4) });
    auto dw = denorm_deltas.index({ Slice(), Slice(2, None, 4) });
    auto dh = denorm_deltas.index({ Slice(), Slice(3, None, 4) });

    float max_ratio = abs(log(wh_ratio_clip));
    dw = dw.clamp(-max_ratio, max_ratio);
    dh = dh.clamp(-max_ratio, max_ratio);

    // Compute center of each roi
    auto px = ((rois.index({ Slice(), 0 }) + rois.index({ Slice(), 2 })) * 0.5).unsqueeze(1).expand_as(dx);
    auto py = ((rois.index({ Slice(), 1 }) + rois.index({ Slice(), 3 })) * 0.5).unsqueeze(1).expand_as(dy);
    // Compute width / height of each roi
    auto pw = (rois.index({ Slice(), 2 }) - rois.index({ Slice(), 0 })).unsqueeze(1).expand_as(dw);
    auto ph = (rois.index({ Slice(), 3 }) - rois.index({ Slice(), 1 })).unsqueeze(1).expand_as(dh);
    // Use exp(network energy) to enlarge / shrink each roi
    auto gw = pw * dw.exp();
    auto gh = ph * dh.exp();
    // Use network energy to shift the center of each roi
    auto gx = px + pw * dx;
    auto gy = py + ph * dy;
    // Convert center - xy / width / height to top - left, bottom - right
    auto x1 = gx - gw * 0.5;
    auto y1 = gy - gh * 0.5;
    auto x2 = gx + gw * 0.5;
    auto y2 = gy + gh * 0.5;

    x1 = x1.clamp(0, (int64_t)std::get<1>(img_shape));
    y1 = y1.clamp(0, (int64_t)std::get<0>(img_shape));
    x2 = x2.clamp(0, (int64_t)std::get<1>(img_shape));
    y2 = y2.clamp(0, (int64_t)std::get<0>(img_shape));
    return torch::stack({ x1, y1, x2, y2 }, -1).view_as(deltas);
}

class AnchorGenerator {
public:
    AnchorGenerator(vector<int>& strides, torch::Device device) :device(device) {
        // assuming scale_major = true
        for (auto stride : strides) {
            this->strides.push_back(std::tuple<int, int>(stride, stride));
            base_sizes.push_back(stride);
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        this->ratios = torch::tensor({ 0.5, 1.0, 2.0 }, options).to(device);

        this->scales = torch::tensor({ 8.0 }, options).to(device);
        this->base_anchors = gen_base_anchors(base_sizes, scales, ratios);

        //RPN bbox
        this->means = torch::tensor({ 0., 0., 0., 0. }, options).to(device);
        this->stds = torch::tensor({ 1., 1., 1., 1. }, options).to(device);
    }

    TensorVector grid_anchors(vector<std::tuple<int64_t, int64_t>>& featmap_sizes) {
        TensorVector multi_level_anchors(featmap_sizes.size());
        for (size_t i = 0; i < featmap_sizes.size(); i++) {
            auto featmap_size = featmap_sizes[i];
            auto base_anchor = this->base_anchors[i];

            auto stride = this->strides[i];

            auto feat_h = std::get<0>(featmap_size);
            auto feat_w = std::get<1>(featmap_size);

            auto shift_x = torch::arange(0, feat_w, device = device) * std::get<0>(stride);
            auto shift_y = torch::arange(0, feat_h, device = device) * std::get<0>(stride);
            //assuming row_major=True
            auto shift_xx = shift_x.repeat((shift_y.size(0)));
            auto shift_yy = shift_y.view({ -1, 1 }).repeat({ 1, shift_x.size(0) }).view(-1);

            auto shifts = torch::stack({ shift_xx, shift_yy, shift_xx, shift_yy }, -1);
            shifts = shifts.type_as(base_anchor);

            auto all_anchors_i = base_anchor.index({ None, Slice(), Slice() }) +
                shifts.index({ Slice(), None, Slice() });
            multi_level_anchors[i] = all_anchors_i.view({ -1, 4 });
        }
        return multi_level_anchors;
    }

    TensorVector get_bboxes(TensorVector cls_scores, TensorVector bbox_preds, vector<ImgMeta> img_metas) {
        auto num_levels = cls_scores.size();
        vector<std::tuple<int64_t, int64_t>> featmap_sizes(num_levels);

        int64_t dim_num = cls_scores[0].dim();
        for (size_t i = 0; i < num_levels; i++) {
            auto& cls_score = cls_scores[i];
            featmap_sizes[i] = std::make_tuple<int64_t, int64_t>(cls_score.size(dim_num - 2), cls_score.size(dim_num - 1));
        }

        auto mlvl_anchors = this->grid_anchors(featmap_sizes);

        TensorVector cls_score_list(num_levels), bbox_pred_list(num_levels);
        TensorVector result_list(img_metas.size());
        for (size_t i = 0; i < img_metas.size(); i++) {
            for (size_t j = 0; j < num_levels; j++) {
                cls_score_list[j] = cls_scores[j].index({ (int64_t)i });
                bbox_pred_list[j] = bbox_preds[j].index({ (int64_t)i });
            }
            auto proposals = this->_get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors,
                img_metas[i].img_shape);
            result_list[i] = proposals;
        }
        return result_list;
    }

private:
    vector<std::tuple<int, int>> strides;
    vector<int> base_sizes;
    TensorVector base_anchors;
    Tensor ratios, scales;
    torch::Device device;
    Tensor means, stds;

    TensorVector gen_base_anchors(vector<int>& base_sizes,
        Tensor scales, Tensor ratios) {
        TensorVector base_anchors;
        for (int base_size : base_sizes) {
            int w = base_size, h = base_size;
            auto h_ratios = torch::sqrt(ratios);
            auto w_ratios = 1 / h_ratios;

            auto ws = (w * w_ratios.index({ Slice(), None }) * scales.index({ None, Slice() })).view(-1);
            auto hs = (h * h_ratios.index({ Slice(), None }) * scales.index({ None, Slice() })).view(-1);
            base_anchors.push_back(torch::stack({ -0.5 * ws, -0.5 * hs , 0.5 * ws , 0.5 * hs }, -1).to(device));
        }
        return base_anchors;
    }

    Tensor _get_bboxes_single(TensorVector cls_score_list, TensorVector bbox_pred_list,
        TensorVector mlvl_anchors, std::tuple<size_t, size_t, size_t> img_shape) {
        auto num_levels = cls_score_list.size();
        int64_t nms_pre = 1000;

        TensorVector mlvl_scores(num_levels), mlvl_bbox_preds(num_levels);
        TensorVector level_ids(num_levels), mlvl_valid_anchors(num_levels);

        for (size_t i = 0; i < num_levels; i++) {
            auto rpn_cls_score = cls_score_list[i];
            auto rpn_bbox_pred = bbox_pred_list[i];

            rpn_cls_score = rpn_cls_score.permute({ 1, 2, 0 });
            rpn_cls_score = rpn_cls_score.reshape(-1);
            auto scores = rpn_cls_score.sigmoid();

            rpn_bbox_pred = rpn_bbox_pred.permute({ 1, 2, 0 }).reshape({ -1, 4 });
            auto anchors = mlvl_anchors[i];

            if (scores.size(0) > nms_pre) {
                //ranked_scores, rank_inds = scores.sort(-1, true)
                auto rets = scores.sort(-1, true);
                auto ranked_scores = std::get<0>(rets);
                auto rank_inds = std::get<1>(rets);
                auto topk_inds = rank_inds.index({ Slice(None, nms_pre) });
                scores = ranked_scores.index({ Slice(None, nms_pre) });
                rpn_bbox_pred = rpn_bbox_pred.index({ topk_inds , Slice() });
                anchors = anchors.index({ topk_inds , Slice() });
            }
            mlvl_scores[i] = scores;
            mlvl_bbox_preds[i] = rpn_bbox_pred;
            mlvl_valid_anchors[i] = anchors;
            level_ids[i] =
                scores.new_full({ scores.size(0) }, (int64_t)i, torch::TensorOptions().dtype(torch::kLong));
        }

        Tensor scores = torch::cat(mlvl_scores);
        Tensor anchors = torch::cat(mlvl_valid_anchors);
        Tensor rpn_bbox_pred = torch::cat(mlvl_bbox_preds);


        auto proposals = delta2bbox(anchors, rpn_bbox_pred, img_shape, this->means, this->stds);
        auto ids = torch::cat(level_ids);

        auto output = batched_nms(proposals, scores, ids, false, 0.7);
        return std::get<0>(output).index({ Slice(None, 1000) });
    }
};
class ROIAlignFunction {
public:
    ROIAlignFunction(std::tuple<int, int> out_size, float spatial_scale) :out_size(out_size), spatial_scale(spatial_scale) {}
    Tensor forward(Tensor features, Tensor rois) {
        return ROIAlign_forwardV2(features, rois, this->spatial_scale,
            std::get<0>(this->out_size), std::get<1>(this->out_size), 0,
            true);
    }
private:
    std::tuple<int, int> out_size;
    float spatial_scale;
};

class ROIHEAD {
public:
    ROIHEAD(string model_path, std::tuple<int, int> out_size, vector<int> featmap_strides, int featmap_num,
        int num_classes, torch::Device device) : device(device), out_size(out_size),
        featmap_num(featmap_num), num_classes(num_classes) {
        for (size_t i = 0; i < featmap_num; i++) {
            this->roi_layers.push_back(ROIAlignFunction(out_size, 1.0 / (float)featmap_strides[i]));
        }
        this->roi_head_model = torch::jit::load(model_path);
        this->roi_head_model.to(device);

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        //RPN bbox
        this->means = torch::tensor({ 0., 0., 0., 0. }, options).to(device);
        this->stds = torch::tensor({ 0.1, 0.1, 0.2, 0.2 }, options).to(device);

    }
    vector<vector<COORD>> simple_test(TensorVector x, TensorVector proposal_list, ImgMeta img_meta) {
        //合并simple_test_bboxes 和 simple_test
        auto rois = this->bbox2roi(proposal_list);

        //{cls_score, bbox_pred}
        auto output = _bbox_forward(x, rois);
        output = this->get_bboxes(rois, std::get<0>(output), std::get<1>(output),
            img_meta.img_shape, img_meta.scale_factor, true);

        //det_bboxes, det_labels = output
        //bbox2result
        auto det_bboxes = std::get<0>(output);
        auto det_labels = std::get<1>(output);

        // debug
        // torch::jit::script::Module results = torch::jit::load("../results.pt");
        // std::cout << "det rets" << std::endl;
        // auto std_det_bboxes = results.attr("det_bboxes").toTensor();
        // auto std_det_labels = results.attr("det_labels").toTensor();
        // std::cout << torch::sum(std_det_bboxes - det_bboxes) << std::endl;
        // std::cout << torch::sum(std_det_labels - det_labels) << std::endl;

        vector<vector<COORD>> result(this->num_classes);
        for (int i = 0; i < this->num_classes; i++)
            result[i] = vector<COORD>();

        if (det_bboxes.size(0) == 0) {
            return result;
        }
        else {
            det_bboxes.to(torch::kCPU);
            det_labels.to(torch::kCPU);
            for (int64_t i = 0; i < det_labels.size(0); i++) {
                int64_t label = det_labels.index({ i }).item<int64_t>();
                auto bbox = det_bboxes.index({ i });
                float x1 = bbox.index({ 0 }).item<float>();
                float y1 = bbox.index({ 1 }).item<float>();
                float x2 = bbox.index({ 2 }).item<float>();
                float y2 = bbox.index({ 3 }).item<float>();
                float score = bbox.index({ 4 }).item<float>();

                result[label].push_back(std::make_tuple(x1, y1, x2, y2, score));
            }
            return result;
        }

    }
private:
    vector<ROIAlignFunction> roi_layers;
    std::tuple<int, int> out_size;
    int featmap_num;
    torch::jit::script::Module roi_head_model;
    Tensor means, stds;
    torch::Device device;
    int num_classes;

    Tensor bbox2roi(TensorVector bbox_list) {
        TensorVector rois_list(bbox_list.size());
        for (size_t i = 0; i < bbox_list.size(); i++) {
            auto bboxes = bbox_list[i];
            Tensor rois;
            if (bboxes.size(0) > 0) {
                auto img_inds = bboxes.new_full({ bboxes.size(0), 1 }, (int64_t)i);
                rois = torch::cat({ img_inds, bboxes.index({Slice(), Slice(None, 4)}) }, -1);
            }
            else {
                rois = bboxes.new_zeros((0, 5));
            }
            rois_list[i] = rois;
        }
        return torch::cat(rois_list, 0);
    }

    std::tuple<Tensor, Tensor> _bbox_forward(TensorVector& x, Tensor rois) {
        //注意：只用4个featuremap
        TensorVector::const_iterator first1 = x.begin();
        TensorVector::const_iterator last1 = x.begin() + this->featmap_num;
        auto bbox_feats = bbox_roi_extractor(TensorVector(first1, last1), rois);

        std::vector<torch::jit::IValue> inputs = { bbox_feats };
        auto outputs = this->roi_head_model.forward(inputs).toTuple();

        auto cls_score = outputs->elements()[0].toTensor();
        auto bbox_pred = outputs->elements()[1].toTensor();
        return std::make_tuple(cls_score, bbox_pred);
    }

    Tensor bbox_roi_extractor(TensorVector feats, Tensor rois) {
        // out_size = 7
        int out_size = 7, num_levels = feats.size(), out_channels = 256;
        auto roi_feats = feats[0].new_zeros(
            { rois.size(0), out_channels, out_size, out_size });
        auto target_lvls = this->map_roi_levels(rois, num_levels);

        for (size_t i = 0; i < num_levels; i++) {
            auto inds = (target_lvls == (int64_t)i);
            if (inds.any().item<bool>()) {
                auto rois_ = rois.index({ inds, Slice() });
                auto roi_feats_t = this->roi_layers[i].forward(feats[i], rois_);
                roi_feats.index_put_({ inds }, roi_feats_t);
            }
            //else分支没有实现，理论上没有影响
        }
        return roi_feats;
    }

    Tensor map_roi_levels(Tensor rois, int num_levels) {
        /*Map rois to corresponding feature levels by scales.

            - scale < finest_scale * 2: level 0
            - finest_scale * 2 <= scale < finest_scale * 4 : level 1
            - finest_scale * 4 <= scale < finest_scale * 8 : level 2
            - scale >= finest_scale * 8 : level 3
        */
        auto scale = torch::sqrt(
            (rois.index({ Slice(), 3 }) - rois.index({ Slice(), 1 })) *
            (rois.index({ Slice(), 4 }) - rois.index({ Slice(), 2 })));
        auto target_lvls = torch::floor(torch::log2(scale / 56 + 1e-6));
        target_lvls = target_lvls.clamp(0, num_levels - 1).to(torch::kLong);
        return target_lvls;
    }
    std::tuple<Tensor, Tensor> get_bboxes(Tensor rois, Tensor cls_score, Tensor bbox_pred,
        std::tuple<size_t, size_t, size_t>img_shape,
        Tensor scale_factor, bool rescale) {
        auto scores = torch::nn::functional::softmax(cls_score, 1);
        auto bboxes = delta2bbox(rois.index({ Slice(), Slice(1, None) }), bbox_pred, img_shape, this->means, this->stds);
        //rescale = true
        bboxes = (bboxes.view({ bboxes.size(0), -1, 4 }) / scale_factor).view({ bboxes.size(0), -1 });

        return multiclass_nms(bboxes, scores, 0.05, 0.5, 100);
    }
    std::tuple<Tensor, Tensor> multiclass_nms(Tensor multi_bboxes, Tensor multi_scores, float score_thr,
        float iou_thr, int max_num) {
        int num_classes = multi_scores.size(1) - 1;
        Tensor bboxes;
        if (multi_bboxes.size(1) > 4) {
            bboxes = multi_bboxes.view({ multi_scores.size(0), -1, 4 });
        }
        else {
            bboxes = multi_bboxes.index({ Slice(), None }).expand({ -1, num_classes, 4 });
        }
        auto scores = multi_scores.index({ Slice(), Slice(None, -1) });

        auto valid_mask = scores > score_thr;
        bboxes = bboxes.index({ valid_mask });
        scores = scores.index({ valid_mask });
        auto labels = valid_mask.nonzero().index({ Slice(), 1 });

        if (bboxes.numel() == 0) {
            bboxes = multi_bboxes.new_zeros({ 0, 5 });
            labels = multi_bboxes.new_zeros({ 0 }, torch::TensorOptions().dtype(torch::kLong));
            return std::make_tuple(bboxes, labels);
        }

        // bboxes = torch::cat({ bboxes, scores.index({Slice(), None}) }, -1);
        auto output = batched_nms(bboxes, scores, labels, false, 0.5);
        auto dets = std::get<0>(output);
        auto keep = std::get<1>(output);
        if (max_num > 0) {
            dets = dets.index({ Slice(None, max_num) });
            keep = keep.index({ Slice(None, max_num) });
        }
        return std::make_tuple(dets, labels.index({ keep }));
    }
};

class ImageReader {
public:
    ImageReader(bool to_rgb, torch::Device device) :device(device) {
        this->img_scale = std::make_tuple(1333, 800);
        this->max_long_edge = std::max(std::get<0>(this->img_scale), std::get<1>(this->img_scale));
        this->max_short_edge = std::min(std::get<0>(this->img_scale), std::get<1>(this->img_scale));
        this->options = torch::TensorOptions().dtype(torch::kFloat);

        this->mean_ori = torch::tensor({ 123.675, 116.28 , 103.53 }, options).to(this->device);
        this->std_ori = torch::tensor({ 58.395, 57.12 , 57.375 }, options).to(this->device);

        this->mean = mean_ori.reshape({ 1, -1 }).to(this->device);
        this->stdinv = 1.0 / std_ori.reshape({ 1, -1 }).to(this->device);

        this->to_rgb = to_rgb;

        this->size_divisor = 32;
    }
    std::tuple<Tensor, ImgMeta> imread(string& file_name) {
        cv::Mat img_mat;
        ImgMeta img_meta;
        img_meta.file_name = file_name;
        img_mat = cv::imread(file_name, cv::IMREAD_COLOR);
        if (!img_mat.data) {
            return std::tuple<Tensor, ImgMeta>(torch::zeros({ 0 }, this->options).to(this->device), img_meta);
        }
        int rows = img_mat.rows, cols = img_mat.cols, c = img_mat.channels();
        assert(c == 3);
        auto img = transform(img_mat, img_meta);
        return std::make_tuple(img, img_meta);
    }
    Tensor transform(cv::Mat& img_mat, ImgMeta& img_meta) {
        int old_w = img_mat.cols, old_h = img_mat.rows;
        float scale_factor = std::min((float)max_long_edge / (float)std::max(old_w, old_h),
            (float)max_short_edge / (float)std::min(old_w, old_h));

        int new_w = (int)(old_w * scale_factor + 0.5);
        int new_h = (int)(old_h * scale_factor + 0.5);
        img_meta.img_shape = std::make_tuple(new_h, new_w, 3);

        float w_scale = (float)new_w / (float)old_w, h_scale = (float)new_h / (float)old_h;
        img_meta.scale_factor = torch::tensor({ w_scale, h_scale, w_scale, h_scale }, this->options).to(this->device);

        cv::resize(img_mat, img_mat, cv::Size(new_w, new_h), cv::INTER_LINEAR);
        if (this->to_rgb) {
            cv::cvtColor(img_mat, img_mat, cv::COLOR_BGR2RGB);
        }
        img_mat.convertTo(img_mat, CV_32FC3);

        auto img = torch::from_blob(img_mat.data, { new_h, new_w, 3 }, this->options).clone().to(this->device);
        img.sub_(this->mean);
        img.mul_(this->stdinv);

        //zero padding
        auto pad_h = int(std::ceil((float)new_h / (float)this->size_divisor)) * this->size_divisor;
        auto pad_w = int(std::ceil((float)new_w / (float)this->size_divisor)) * this->size_divisor;
        img = img.unsqueeze_(0).permute({ 0, 3, 1, 2 });

        img = torch::nn::functional::pad(img, torch::nn::functional::PadFuncOptions({ 0, pad_w - new_w,
            0, pad_h - new_h }));

        return img.contiguous();
    }

private:
    std::tuple<int, int> img_scale;
    int max_long_edge, max_short_edge;
    torch::TensorOptions options;
    torch::Device device;
    Tensor mean_ori, std_ori, mean, stdinv;
    int size_divisor;
    bool to_rgb;
};


int main() {
    torch::NoGradGuard no_grad;
    torch::Device device("cuda:0");
    vector<int> strides = vector<int>({ 4,8,16,32,64 });
    ImageReader img_reader(true, device);
    torch::jit::script::Module FasterRcnn = torch::jit::load("../FasterRcnn_cpp.pt");
    FasterRcnn.to(device);
    AnchorGenerator anchor_gen(strides, device);
    string cnn_model_path = "../ROIHead_cpp.pt";
    ROIHEAD roi_head(cnn_model_path, std::make_tuple<int, int>(7, 7), strides, 4, 80, device);
    //加载python的输出结果，用于debug
    //torch::jit::script::Module results = torch::jit::load("../results.pt");

    string images_path;
    std::cin >> images_path;

    vector<std::tuple<string, string>> file_list = listdir(images_path);
    //std::cout << file_list.size() << std::endl;
    json all_output;
    int64_t step = 50, cnt = 0;
    auto start = std::chrono::system_clock::now();
    for (auto img_info : file_list) {
        cnt += 1;
        if (cnt % step == 0){
            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            auto time_s = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
            std::cout << "step: " << cnt << "/" << file_list.size() << std::endl;
            std::cout << "time: "<< time_s / (double)step << "s" <<std::endl;
            // t0 = clock();
            //std::cout << "time: "<< (double)t_acc / (double)step / CLOCKS_PER_SEC << "s" <<std::endl;
            //t_acc = 0;
            start = std::chrono::system_clock::now();
            
        }
        auto img_path = std::get<0>(img_info);
        auto img_name = std::get<1>(img_info);

        auto img_output = img_reader.imread(img_path);
        Tensor img = std::get<0>(img_output);
        ImgMeta img_meta = std::get<1>(img_output);

        if (!img.size(0)) {
            std::cout << "cannot read file" << std::endl;
        }

        //TensorVector std_mlvl_anchors = results.attr("mlvl_anchors").toTensorVector();
        //std::cout << img_path << std::endl;
        //Tensor std_img = results.attr("img").toTensor().to(device);
        //std::cout << torch::sum(torch::abs(std_img - img)) << std::endl;
        //std::cout << torch::sum(torch::abs(std_img.to(device) - img)) << std::endl;
        //img_meta.img_shape = std::make_tuple(800, 1280, 3);
        //img_meta.scale_factor = torch::tensor({1.509434, 1.509434, 1.509434, 1.509434}, std_img.options()).to(device);

        vector<ImgMeta> img_metas = { img_meta };

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img);

        //t0 = clock();

        auto outputs = FasterRcnn.forward(inputs).toTuple();
        //5个元素
        auto feature_maps = outputs->elements()[0].toTuple();
        auto cls_out = outputs->elements()[1].toTensorVector();
        auto bbox_out = outputs->elements()[2].toTensorVector();

        vector<std::tuple<int64_t, int64_t>> featmap_sizes(cls_out.size());
        auto dim_num = cls_out[0].dim();
        for (size_t i = 0; i < cls_out.size(); i++) {
            featmap_sizes[i] = std::make_tuple<int64_t, int64_t>(cls_out[i].size(dim_num - 2), cls_out[i].size(dim_num - 1));
        }

        auto mlvl_anchors = anchor_gen.grid_anchors(featmap_sizes);
        auto proposal_list = anchor_gen.get_bboxes(cls_out, bbox_out, img_metas);

        //roi_head
        //vector<int> strides = vector<int>({ 4,8,16,32,64 });
        //ROIHEAD roi_head = ROIHEAD(std::make_tuple<int, int>(7, 7), strides, 4);
        //torch::jit::script::Module results = torch::jit::load("results.pt");
        //TensorVector std_proposal_list = results.attr("proposal_list").toTensorVector();
        //TensorVector x = results.attr("x").toTensorVector();
        TensorVector x(5);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = feature_maps->elements()[i].toTensor();
        }
        auto detect_rets = roi_head.simple_test(x, proposal_list, img_metas[0]);

        
        json output = json::array();
        for (int cat_id = 0; cat_id < detect_rets.size(); cat_id++) {
            for (int i = 0; i < detect_rets[cat_id].size(); i++) {
                COORD coord = detect_rets[cat_id][i];
                float x1 = std::get<0>(coord);
                float y1 = std::get<1>(coord);
                float x2 = std::get<2>(coord);
                float y2 = std::get<3>(coord);
                float score = std::get<4>(coord);
                json det;
                json bbox = { x1, y1, x2 - x1, y2 - y1 };
                det.emplace("category_id", cat_id + 1);
                det.emplace("bbox", bbox);
                det.emplace("score", score);
                output.emplace_back(det);
            }
        }
        all_output.emplace(img_name, output);
        //t_acc += clock() - t0;
    }

    string s = all_output.dump();
    //std::cout << s << std::endl;
    return 0;
}
