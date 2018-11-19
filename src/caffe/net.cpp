#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

//save file for debugging
#include <fstream>

#include "hdf5.h"
//for random number generator
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/algorithm/string.hpp>
#include "boost/lexical_cast.hpp"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#define round_to_nearest
//#define GPU_ERROR_INJECTION
#define SEARCH_SCHEME 2 //searching scheme, 2 faster but not accurate as it only inject error to the final layer

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  //move this to class lvel variable, to be referenced later NetParameter param;
  NetParameter param;
  InsertSplits(filtered_param, &param);
  net_param_ = param;// move this to class lvel variable, to be referenced later

  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  //  LOG(INFO) << "after forward size "<<after_forward_.size();
    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

//add in new functions for precision analysis

//12 May
template <typename Dtype>
void Net<Dtype>::New_Analyze_Batch(Dtype delta) {
  //new cleaned analyze function with automatic mode, the legacy version is still kept for reference.

  std::cout <<"\n ------------- blobnames before ------- \n";
  vector<string> names = blob_names();
  for(std::vector<int>::size_type i = 0; i != names.size(); i++) {
    std::cout <<names[i] <<" ";
  }

const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer
const int INPUT_LAYER = 2;

//const int OUTPUT_LAYER = layers_.size() - 3;// cifar
//const int INPUT_LAYER = 0;//cifar


std::cout <<"\n ------------- Layersize "<<layers_.size()<<"\n";
vector<string> layernames = layer_names();
for(std::vector<int>::size_type i = 0; i != layernames.size(); i++) {
  std::cout << i << " : "<<layernames[i] <<" \t";
}
std::cout <<"\n";
//output batch counts :


std::vector<float> mac_count;
std::vector<float> weight_count;
std::vector<float> input_count;
int analyzing_layers = 0;
for (int i=0; i<OUTPUT_LAYER; i++){
  if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
  {
      // bottom_vecs_[i][0] input
      //layers_[i]->blobs()[0] weight
      // top_vecs_[i][0] output
    analyzing_layers++;
    weight_count.push_back(layers_[i]->blobs()[0]->count());
    input_count.push_back(bottom_vecs_[i][0]->count());
    std::cout<<"catching layer "<< i <<" type " << layers_[i]->type()<< " shape bottom "<<bottom_vecs_[i][0]->shape_string() <<"\n";
    std::cout<<"catching layer "<< i << " shape top "<<top_vecs_[i][0]->shape_string() <<"\n";
    std::cout<<"catching layer "<< i << " shape weight "<<layers_[i]->blobs()[0]->shape_string() <<"\n";
    if(strcmp (layers_[i]->type(),"InnerProduct") == 0 ){
        mac_count.push_back(layers_[i]->blobs()[0]->count()); //FC layers mac_count = weight_count
    }
    else{ //convolutional layer, shape size must be 4
        //top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3) * sizeof(weight)
        mac_count.push_back(top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3)*layers_[i]->blobs()[0]->count());
    }
  }

}

for(int i = 0; i <analyzing_layers; i++)
  std::cout<<"mac count: "<<mac_count[i]<< "\nweight count: "<<weight_count[i]<<"\n";

std::cout<<"\n input count \n";
for(int i = 0; i <analyzing_layers; i++)
  std::cout<<input_count[i]<<", ";
std::cout<<"\n\n";

std::cout<<"\n mac count \n";
for(int i = 0; i <analyzing_layers; i++)
  std::cout<<mac_count[i]<<", ";
std::cout<<"\n\n";

//for each batch, do the same

   AnalyzeFromToForward(0, layers_.size() - 1); //from 0, do once only

  Dtype* out_array_ref = new Dtype[top_vecs_[OUTPUT_LAYER][0]->count()]();
  //Can use caffe_copy from math_fuctions.hpp here.
  { //debugging block, dont' touch me
    const Dtype* out_array_simulated =  top_vecs_[OUTPUT_LAYER][0]->cpu_data();
    std::ofstream myfile;
    myfile.open ("/home/minh/github/caffe/debugging/out_ref.txt");
    for(int j =0; j < top_vecs_[OUTPUT_LAYER][0]->count(); j ++ ){
      myfile << out_array_simulated[j]<<",";
      out_array_ref[j] = out_array_simulated[j];
    }
    myfile.close();
  }


  const vector<Blob<Dtype>*> result = net_output_blobs_;
  std::cout<<"result size "<<result.size()<<"\n";
  //result [0] is accuracy ; result [1] is loss; care about acc only
  const Dtype* result_vec = result[0]->cpu_data();
  float original_acc = result_vec[0];
  std::cout<<" \n---------------------------------- \n";
  std::cout<<"original accuracy "<<original_acc<<"\n";
  std::cout<<" \n---------------------------------- \n";



  std::cout<<"\n ----- doing forward with error -----\n";

  float current_error = 1.0;
//  float current_error = 0.414; //googlenet 1% rel
//    float current_error = 0.297; //squeezenet
//    float current_error = 0.4375; //squeezenet
//float current_error = 0.34; //alex
//  float current_error = 1.37; //nin 1% rel
//  float current_error = 0.35; //vgg 1% rel
//  float current_error = 0.5; //for plotting linear rel
//  float current_error = 0.492; // resnet50 1% rel
//  float current_error = 0.414; //resnet 152 1% rel
//  float current_error = 1.0; //cifar 1% rel

  float target_sigma = current_error;

  AnalyzeFromToForwardWithError(INPUT_LAYER, layers_.size() - 1, OUTPUT_LAYER, current_error);


  {
    const Dtype* out_array_simulated =  top_vecs_[OUTPUT_LAYER][0]->cpu_data();
    std::ofstream myfile;
    myfile.open ("/home/minh/github/doppio/debugging/out_search.txt");
    for(int j =0; j < top_vecs_[OUTPUT_LAYER][0]->count(); j ++ )
      myfile << out_array_simulated[j]<<",";
    myfile.close();
  }

  {// i want to reuse var name :|
    const vector<Blob<Dtype>*> result_error = net_output_blobs_;
    const Dtype* result_vec_error = result_error[0]->cpu_data();
    float current_acc = result_vec_error[0];
    std::cout<<" \n---------------------------------- \n";
    std::cout<<"current error "<< current_error<<"\n";
    std::cout<<"Simulated Accuracy "<< current_acc<<"\n";
    std::cout<<" \n---------------------------------- \n";
  }



  float my_coefficient = current_error/layers_.size();
  std::cout <<"current _err "<< current_error<< "my coefficient "<< my_coefficient <<"\n";
  //analyzing_layers = N ; define alpha array size N-1. w_portion array size N;

  //int W_vector_count =0;
  //int alpha_vector_count = 0;

   /*
   for (int i =0; i<W_raw.size(); i++){
     W_vector.push_back( boost::lexical_cast<float>(W_raw[i]));
     std::cout<<W_vector.back()<< "\n";
   }*/

   double* sigma_result = new double[analyzing_layers]();

   double* lambda_result = new double[analyzing_layers]();
   double* ck_result = new double[analyzing_layers]();

   int* layer_index = new int[analyzing_layers](); //record down all analyzing layers, no need to use vector

   int layer_index_index = analyzing_layers-1;
   int sigma_result_index = analyzing_layers-1;

   Dtype* rounding_error_out = new Dtype[top_vecs_[OUTPUT_LAYER][0]->count()]();//use to save out_ref - out_simulated

   double temp_sigma_result = my_coefficient;

#define NUM_POINTS 20

   double *coeff = new double[NUM_POINTS]();
   // = { 0.2, 0.4, 0.6,0.8,1.0,1.2,1.4,1.6,1.8,2};
   //range from 0.5 -> 2 / 50 =
   for (int k = 0 ; k < NUM_POINTS; k++){
     coeff[k] = 0.1+k*0.1;
   }
   double std_out;

   /* debugging block
   std::ofstream x_file;
   x_file.open ("/home/minh/github/doppio/debugging/x_file.txt");
   std::ofstream y_file;
   y_file.open ("/home/minh/github/doppio/debugging/y_file.txt");
*/
   double *x_val = new double[NUM_POINTS](); //reuse for multiple layers
   double *y_val = new double[NUM_POINTS](); //reuse for multiple layers

  for (int i=OUTPUT_LAYER; i>0; i--){
    if (strcmp (layers_[i]->type(),"InnerProduct") == 0 || strcmp (layers_[i]->type(),"Convolution") == 0){
        //if any out-of-bound error happens here means something is wrong with network architecture/ not supported
        std::cout <<"analyzing layer "<<i <<"\n";
        //skip the first one
        AnalyzeFromToForwardWithError(i,layers_.size() - 1,i,temp_sigma_result,true);
        const Dtype* out_array = top_vecs_[OUTPUT_LAYER][0]->cpu_data();
        caffe_sub(top_vecs_[OUTPUT_LAYER][0]->count(), out_array_ref, out_array, rounding_error_out); //y = a-b ?
        std_out = CalculateSTD(rounding_error_out,top_vecs_[OUTPUT_LAYER][0]->count());

    //    std::cout<<"["<<std_out<<","<<temp_sigma_result<<"]"<<",";
        temp_sigma_result =current_error*temp_sigma_result/std_out;


    for (int j = 0; j < NUM_POINTS; j++){
        AnalyzeFromToForwardWithError(i,layers_.size() - 1,i,temp_sigma_result,true);

        const Dtype* out_array = top_vecs_[OUTPUT_LAYER][0]->cpu_data();
        caffe_sub(top_vecs_[OUTPUT_LAYER][0]->count(), out_array_ref, out_array, rounding_error_out); //y = a-b ?
        std_out = CalculateSTD(rounding_error_out,top_vecs_[OUTPUT_LAYER][0]->count());
        std::cout<<"["<<std_out<<","<<temp_sigma_result<<"]"<<",";
        //x_file<<std_out<<",";
        //y_file<<temp_sigma_result<<",";
        x_val[j] = std_out;
        y_val[j] = temp_sigma_result;
        temp_sigma_result =current_error*coeff[j]*temp_sigma_result/std_out;
        //std::cout<<"sigma result " << temp_sigma_result<<"\n";
      }


        double lambda = 0;
        double ck = 0;
        linear_regression(NUM_POINTS, x_val, y_val, &lambda, &ck);
        std::cout<<"\n lambda"<<lambda<<" CK " <<ck<<"\n";
        lambda_result[sigma_result_index] = lambda;
        ck_result[sigma_result_index] = ck;

        sigma_result[sigma_result_index] = lambda * current_error + ck;
      //  sigma_result[sigma_result_index] = temp_sigma_result;
      //  std::cout<<sigma_result_index<<"\n";
        sigma_result_index -- ;
      //  std::cout<<layer_index_index<<"\n";
        layer_index[layer_index_index] = i;
        layer_index_index --;

    }// end if layer type = conv or fc

  } //end for over all layer

  //x_file.close();
  //y_file.close();



  std::ofstream output_file;
  output_file.open ("debugging/lambda_theta.txt");

  std::cout << "\n lambda result \n";
  for (int i =0 ; i<analyzing_layers; i++){
    std::cout <<lambda_result[i]<<",";
  }
  std::cout<<"\n";
  std::cout << "\n ck result \n";
  for (int i =0 ; i<analyzing_layers; i++){
    std::cout <<ck_result[i]<<",";
  }
  std::cout<<"\n";


  for (int i =0 ; i<analyzing_layers; i++){
    output_file <<lambda_result[i]<<",";
  }
  output_file<<"\n";
  for (int i =0 ; i<analyzing_layers; i++){
    output_file <<ck_result[i]<<",";
  }
  output_file<<"\n";



  std::cout << "\n sigma result \n";
  for (int i =0 ; i<analyzing_layers; i++){
    std::cout <<lambda_result[i]*(sqrt(1.0/analyzing_layers))*target_sigma + ck_result[i]<<",";
  }
  std::cout<<"\n";



}//end of New_Analyze_Batch



template <typename Dtype>
void Net<Dtype>::SearchSigmaOutput(Dtype error_tolerance) {
  //new accuracy drop in number e.g.  0.05 = 5 percent loss
  int scheme = SEARCH_SCHEME;

  std::cout <<"\n ------------- blobnames before ------- \n";
  vector<string> names = blob_names();
  for(std::vector<int>::size_type i = 0; i != names.size(); i++) {
    std::cout <<names[i] <<" ";
  }

//size-3 = without top-5 layer
//size-4 for imagenet's networks
const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer

//const int OUTPUT_LAYER = layers_.size() - 5;
//const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer

std::cout <<"\n ------------- Layersize "<<layers_.size()<<"\n";
vector<string> layernames = layer_names();
for(std::vector<int>::size_type i = 0; i != layernames.size(); i++) {
  std::cout << i << " : "<<layernames[i] <<" \t";
}
std::cout <<"\n";


//float original_acc = 0.689; //googlenet
//float original_acc = 0.68358; //vgg_16
//float original_acc = 0.685; //vgg_19
//float original_acc = 0.728; //resnet-50
//float original_acc = 0.7524; //resnet152
//float original_acc = 0.7114; //cifar10
//float original_acc = 0.56156; //nin
//float original_acc = 0.5768; //squeezenet
//float original_acc = 0.695; // mobilenet
float original_acc = 0.569; // alexnet

std::cout<<"\n ----- doing forward with error -----\n";
std::cout<<"output layer name "<<layer_names_[OUTPUT_LAYER]<<"\n";
//initialize the search procedure
float tolerance = original_acc*1.0/100 ; //1% accuracy loss from the original relatively
float stop_epsilon = 0.01; // stop if upper_bound - lower_bound < stop_epsilon
float lower_bound  = 0;
float upper_bound = 1;
float current_error = upper_bound/2;

int num_batches = 50; //alexnet 500 per batch
//int num_batches = 1000; //vgg
//int num_batches = 500;//nin; //resnet152, 25 per batch

//int num_batches = 200; //googlenet
//const int accuracy_index = 1; //googlenet
const int accuracy_index = 0; //vgg_16

// searching / uncomment in real run
  bool stop_phase1 = false;
  while(!stop_phase1){
    std::cout<<"\nsearching for error in the last layer, current value "<<  current_error
      <<" lower_bound "<< lower_bound << " upper_bound "<<upper_bound<<"\n";



//inside the while loop
//for each batch, do the same

  float loss = 0;
  Dtype current_acc = 0;


  if (scheme == 1){
      ProcessLambdaThetaEqual(upper_bound);
  }

  for (int i = 0 ; i<num_batches; i++){
    //const vector<Blob<Dtype>*>& result =  Forward(&iter_loss);
    if (scheme == 1)
      AnalyzeFromToForwardWithErrorEqualDistributed(0, layers_.size() - 1, upper_bound);
    else
      AnalyzeFromToForwardWithError(0, layers_.size() - 1, OUTPUT_LAYER, upper_bound);
    const vector<Blob<Dtype>*>& result  = net_output_blobs_;
    current_acc += result[accuracy_index]->cpu_data()[0];
    std::cout << "Batch " << i << " " << result[accuracy_index]->cpu_data()[0] <<"\n";
  }

  current_acc /= num_batches;
  std::cout << "accuracy : " << current_acc <<"\n";


  if(original_acc - current_acc > tolerance)
    stop_phase1 = true;
  else
    upper_bound = upper_bound*2;
}


  current_error = upper_bound/2;
  stop_epsilon = upper_bound/40;



  bool stop_condition = false;
  while(!stop_condition){
    std::cout<<"\nsearching for error in the last layer, current value "<<  current_error
      <<" lower_bound "<< lower_bound << " upper_bound "<<upper_bound<<"\n";

//start batch processing
      //inside the while loop
      //for each batch, do the same


        float loss = 0;
        Dtype current_acc = 0;


        if (scheme == 1){
            ProcessLambdaThetaEqual(current_error);
        }

        for (int i = 0 ; i<num_batches; i++){
          //const vector<Blob<Dtype>*>& result =  Forward(&iter_loss);
          if (scheme == 1)
            AnalyzeFromToForwardWithErrorEqualDistributed(0, layers_.size() - 1, current_error);
          else
            AnalyzeFromToForwardWithError(0, layers_.size() - 1, OUTPUT_LAYER, current_error);
          const vector<Blob<Dtype>*>& result  = net_output_blobs_;
          current_acc += result[accuracy_index]->cpu_data()[0];
          std::cout << "Batch " << i << " " << result[accuracy_index]->cpu_data()[0] <<"\n";
        }

        current_acc /= num_batches;
        std::cout << "accuracy : " << current_acc <<"\n";
//end batch processing





//    if ((original_acc - current_acc < tolerance) && (upper_bound - lower_bound < stop_epsilon)){
// no need to check for original_acc - current_acc < tolerance, upper bound can be served as result here
    if ((upper_bound - lower_bound < stop_epsilon)){
      stop_condition = true;
      current_error = upper_bound;
    }
    else{
      if (original_acc - current_acc > tolerance){ //reduce current_error
        upper_bound = current_error;
      }else{
        lower_bound = current_error;
      }
      current_error = (lower_bound + upper_bound)/2;
    }

  } //end while (stop condition)

  std::cout<<"end of searching "<< current_error<<"\n";

  {
    //for measuring time of the search
    std::ofstream myfile;
    myfile.open ("debugging/end_of_search.txt",std::ofstream::app);
    myfile <<"dummy text";
    myfile.close();
  }
  //AnalyzeFromToForwardWithError(0, layers_.size() - 1, OUTPUT_LAYER, current_error);



}//end of Search sigma batches

//for analysis, not used normally.
template <typename Dtype>
void Net<Dtype>::StatsSigmaAcc() {
  //new accuracy drop in number e.g.  0.05 = 5 percent loss


//size-3 = without top-5 layer
//size-4 for imagenet's networks
const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer
//const int OUTPUT_LAYER = layers_.size() - 5;
//const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer

//float original_acc = 0.689; //googlenet
//float original_acc = 0.68358; //vgg_16
//float original_acc = 0.685; //vgg_19
//float original_acc = 0.728; //resnet-50
//float original_acc = 0.7524; //resnet152
//float original_acc = 0.7114; //cifar10
//float original_acc = 0.56156; //nin
//float original_acc = 0.5768; //squeezenet
//float original_acc = 0.695; // mobilenet
float original_acc = 0.569; // alexnet

std::cout<<"\n ----- doing forward with error -----\n";
std::cout<<"output layer name "<<layer_names_[OUTPUT_LAYER]<<"\n";


int num_batches = 100; //vgg
//int num_batches = 1000; //vgg
//int num_batches = 500;//nin; //resnet152, 25 per batch

//int num_batches = 200; //googlenet
//const int accuracy_index = 1; //googlenet
const int accuracy_index = 0; //vgg_16
float sigma = 1.0;
// searching / uncomment in real run
//inside the while loop
//for each batch, do the same
#undef NUM_POINTS
#define NUM_POINTS 1
double *x_val = new double[NUM_POINTS]();
double *y_val = new double[NUM_POINTS]();
  for(int k = 0; k < NUM_POINTS; k ++){
//      sigma  = 0.1 + 0.1*k;
      float loss = 0;
      Dtype current_acc = 0;
      for (int i = 0 ; i<num_batches; i++){
        //const vector<Blob<Dtype>*>& result =  Forward(&iter_loss);
        AnalyzeFromToForwardWithError(0, layers_.size() - 1, OUTPUT_LAYER, sigma);
        const vector<Blob<Dtype>*>& result  = net_output_blobs_;
        current_acc += result[accuracy_index]->cpu_data()[0];
      //  std::cout << "Batch " << i << " " << result[accuracy_index]->cpu_data()[0] <<"\n";
      }
      current_acc /= num_batches;
      std::cout << "sigma" << sigma << "accuracy : " << current_acc <<"\n";
      x_val[k] = sigma;
      y_val[k] = current_acc;
  }
  //print x_y


}//end of StatsSigmaAcc




template <typename Dtype>
void Net<Dtype>::TestSigma() {


  //size-3 = without top-5 layer
  //size-4 for imagenet's networks
  const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer
  //const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer
  const int INPUT_LAYER = 2; //cifar

  std::vector<float> mac_count;
  std::vector<float> weight_count;
  std::vector<float> input_count;
  std::vector<int> layer_index;
  int analyzing_layers = 0;
  for (int i=0; i<=OUTPUT_LAYER; i++){
    if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
    {
        // bottom_vecs_[i][0] input
        //layers_[i]->blobs()[0] weight
        // top_vecs_[i][0] output
      analyzing_layers++;
      weight_count.push_back(layers_[i]->blobs()[0]->count());
      input_count.push_back(bottom_vecs_[i][0]->count());
      if(strcmp (layers_[i]->type(),"InnerProduct") == 0 ){
          mac_count.push_back(layers_[i]->blobs()[0]->count()); //FC layers mac_count = weight_count
      }
      else{ //convolutional layer, shape size must be 4
          //top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3) * sizeof(weight)
          mac_count.push_back(top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3)*layers_[i]->blobs()[0]->count());
      }
      layer_index.push_back(i);
    }

  }

//getting optimized result
std::vector<float> sigma_result;

int sigma_vector_count = 0;

vector<string> sigma_raw;
string temp_line;

 std::ifstream solution("debugging/sigma_solution.txt");
 std::getline(solution,temp_line);
 boost::split(sigma_raw,temp_line,boost::is_any_of(","));
 if((sigma_raw.size() != analyzing_layers) ){
    std::cout<<"wrong solution file, expecting a file with "<< analyzing_layers << " numbers on the first line\n";
  }
 for (int i =0; i<sigma_raw.size(); i++){
   std::cout<<sigma_raw[i]<<"\n";
   sigma_result.push_back(boost::lexical_cast<float> (sigma_raw[i]));
 }

  std::cout<<"\n sigma result \n";
  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<sigma_result[i]<<", ";
  std::cout<<"\n\n";


std::cout<<"\n ----- doing forward with error -----\n";
std::cout<<"output layer name "<<layer_names_[OUTPUT_LAYER]<<"\n";

int num_batches = 50; //cifar
const int accuracy_index = 0; //googlenet
//const int accuracy_index = 0; //vgg_16-19
//0.7267

float loss = 0;
Dtype current_acc_simulate = 0;
std::vector<float> accuracy_simulate;
int max_analyzing_layer = analyzing_layers;
for (int k = 0 ; k<num_batches; k++){
  //const vector<Blob<Dtype>*>& result =  Forward(&iter_loss);
  ForwardFromTo(0, layers_.size() - 1);

  for (int i=0; i<max_analyzing_layer; i++){
          if(i ==0 ){
            //use this to simulate the behaviour of injecting error
            AnalyzeFromToForwardWithError(INPUT_LAYER, layer_index[i], layer_index[i], sigma_result[i],true );
          }
          else if (i == max_analyzing_layer -1){
            AnalyzeFromToForwardWithError(layer_index[i-1]+1, layers_.size() - 1, layer_index[i], sigma_result[i],true);
          }
        else{
            AnalyzeFromToForwardWithError(layer_index[i-1]+1, layer_index[i],layer_index[i], sigma_result[i], true);
        }

    }

  { // the same dirty trick to reuse var name
    const vector<Blob<Dtype>*>& result  = net_output_blobs_;
    current_acc_simulate += result[accuracy_index]->cpu_data()[0];
    std::cout << "Batch " << k << " " << result[accuracy_index]->cpu_data()[0] <<"\n";

    accuracy_simulate.push_back(result[accuracy_index]->cpu_data()[0]);

  }
} //end loop batches

 //comment fixedpoint section for testing with cifar

current_acc_simulate /= num_batches;
std::cout << "accuracy simulate : " << current_acc_simulate<<"\n";

}//end of TestFixedpoint batches







template <typename Dtype>
void Net<Dtype>::SearchWeightFixedpoint() {


  //size-3 = without top-5 layer
  //size-4 for imagenet's networks
  const int OUTPUT_LAYER = layers_.size() - 4;//  right before softmax. change to custom layer
  //const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer
  const int INPUT_LAYER = 2;
//  const int INPUT_LAYER = 0; //cifar

  std::vector<float> mac_count;
  std::vector<float> weight_count;
  std::vector<float> input_count;
  std::vector<int> layer_index;
  int analyzing_layers = 0;
  for (int i=0; i<=OUTPUT_LAYER; i++){
    if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
    {
        // bottom_vecs_[i][0] input
        //layers_[i]->blobs()[0] weight
        // top_vecs_[i][0] output
      analyzing_layers++;
      weight_count.push_back(layers_[i]->blobs()[0]->count());
      input_count.push_back(bottom_vecs_[i][0]->count());
      if(strcmp (layers_[i]->type(),"InnerProduct") == 0 ){
          mac_count.push_back(layers_[i]->blobs()[0]->count()); //FC layers mac_count = weight_count
      }
      else{ //convolutional layer, shape size must be 4
          //top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3) * sizeof(weight)
          mac_count.push_back(top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3)*layers_[i]->blobs()[0]->count());
      }
      layer_index.push_back(i);
    }

  }

  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<layer_names_[layer_index[i]] <<"mac count: "<<mac_count[i]<< "\nweight count: "<<weight_count[i]<<"\n";

//getting optimized result
std::vector<float> sigma_result;

int sigma_vector_count = 0;

vector<string> sigma_raw;
string temp_line;

 std::ifstream solution("debugging/sigma_solution.txt");
 std::getline(solution,temp_line);
 boost::split(sigma_raw,temp_line,boost::is_any_of(","));
 if((sigma_raw.size() != analyzing_layers) ){
    std::cout<<"wrong solution file, expecting a file with "<< analyzing_layers << " numbers on the first line\n";
    return;
  }
 for (int i =0; i<sigma_raw.size(); i++){
   std::cout<<sigma_raw[i]<<"\n";
   sigma_result.push_back(boost::lexical_cast<float> (sigma_raw[i]));
 }



  std::cout<<"\n input count \n";
  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<input_count[i]<<", ";
  std::cout<<"\n\n";

  std::cout<<"\n sigma result \n";
  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<sigma_result[i]<<", ";
  std::cout<<"\n\n";


float original_acc = 0.689; //googlenet

std::cout<<"\n ----- doing forward with error -----\n";
std::cout<<"output layer name "<<layer_names_[OUTPUT_LAYER]<<"\n";

// imagenet datasize = 50k
// cifar10 datasize = 10k

//int num_batches = 1; //cifar
int num_batches = 50; //vgg batchsize = 100
//int num_batches = 100; //googlenet  batch size =

//int num_batches = 200; //resnet
//const int accuracy_index = 1; //googlenet
const int accuracy_index = 0; //vgg_16-19, cifar, etc
//0.7267
// AnalyzeFromToForward(0, layers_.size() - 1);

int num_remain_batches = 450;//remaining batches to reach datasize and refetching data again.

/*
AnalyzeFromToForwardWithError(INPUT_LAYER, layer_index[0], layer_index[0], sqrt(0.25)*9.82234, true );
AnalyzeFromToForwardWithError(layer_index[0]+1, layer_index[1], layer_index[1], sqrt(0.25)*23.9132, true );
AnalyzeFromToForwardWithError(layer_index[1]+1, layer_index[2], layer_index[2], sqrt(0.25)*37.1326, true );
AnalyzeFromToForwardWithError(layer_index[2]+1, layers_.size() - 1, layer_index[3], sqrt(0.25)*51.8267, true );
*/

int max_analyzing_layer = analyzing_layers;

//ForwardFromTo(0, layers_.size() - 1);
//ForwardFromTo(0, layers_.size() - 1);
//AnalyzeFromToForwardWithError(INPUT_LAYER, layers_.size() - 1, layer_index[0],  sigma_result[0] );

//  AnalyzeFromToForwardFixedpoint(INPUT_LAYER, layers_.size() - 1, layer_index[2], 5,-2, 5,NULL);

//ForwardFromTo(INPUT_LAYER, layers_.size() - 1);
int current_bitwidth_conv = 16;
int current_bitwidth_fc = 16;

float current_error = 0; //relative error
while(current_error <= 0.01){

float loss = 0;
Dtype current_acc_simulate = 0;
Dtype current_acc_fixedpoint = 0;
Dtype current_acc_original = 0;
std::vector<float> accuracy_simulate;
std::vector<float> accuracy_fixedpoint;
std::vector<float> accuracy_original;

for (int k = 0 ; k<num_batches; k++){
  //const vector<Blob<Dtype>*>& result =  Forward(&iter_loss);
  //AnalyzeFromToForwardWithError(0, layers_.size() - 1, OUTPUT_LAYER, upper_bound);
  //if (k!=0 ) // do this first to fetch data & correct the labels
    ForwardFromTo(0, layers_.size() - 1);

    { // the same dirty trick to reuse var name
      const vector<Blob<Dtype>*>& result  = net_output_blobs_;
      current_acc_original += result[accuracy_index]->cpu_data()[0];
      accuracy_original.push_back(result[accuracy_index]->cpu_data()[0]);

    //  for (int j =0;j <result.size(); j++)
    //    std::cout << "Batch " << k << " original " << result[j]->cpu_data()[0] <<"\n";
    }

// doing the same thing with fixedpoint format

  for (int i=0; i<max_analyzing_layer; i++){

    std::pair<int, int*> bitwidth_result = GetBitwidthFromError(sigma_result[i], NULL,layers_[layer_index[i]]->blobs()[0]->count(), true, 0.0);
    //AnalyzeFromToForwardFixedpoint(int start, int end, int layer_error, int a_int,int a_frac, int weight_int,int* weight_frac)
          if(i ==0 ){
            //use this to simulate the behaviour of injecting error
            //no need a_int for now, we don't simulate overflow, simulating dropping least significant bit to compare with stripes instead
            AnalyzeFromToForwardFixedpoint(INPUT_LAYER, layer_index[i], layer_index[i], 5,bitwidth_result.first, 5,current_bitwidth_conv);

          }
          else if (i == max_analyzing_layer -1){
            AnalyzeFromToForwardFixedpoint(layer_index[i-1]+1, layers_.size() - 1, layer_index[i], 5,bitwidth_result.first, 5,current_bitwidth_conv);

          }
        else{
            AnalyzeFromToForwardFixedpoint(layer_index[i-1]+1, layer_index[i], layer_index[i], 5,bitwidth_result.first, 5,current_bitwidth_conv);
        }
    //    bitwidth_result.push_back(GetBitwidthFromError(sigma_result[i], NULL,layers_[layer_index[i]]->blobs()[0]->count(), true, 0.0));
    delete bitwidth_result.second; //delete the weights bitwidth result. conserve mem
  }


  { // the same dirty trick to reuse var name
    const vector<Blob<Dtype>*>& result  = net_output_blobs_;
    current_acc_fixedpoint += result[accuracy_index]->cpu_data()[0];
  //  std::cout << "Batch " << k << " " << result[accuracy_index]->cpu_data()[0] <<"\n";


    accuracy_fixedpoint.push_back(result[accuracy_index]->cpu_data()[0]);

//    for (int j =0;j <result.size(); j++)
//      std::cout << "Batch " << k << " fixedpoint " << result[j]->cpu_data()[0] <<"\n";

  }
 //comment fixedpoint section for testing with cifar

}//end of the loop over batches

//current_acc_simulate /= num_batches;
current_acc_fixedpoint /= num_batches;
current_acc_original /= num_batches;
std::cout << "accuracy original : " << current_acc_original<<"\n";
std::cout << "accuracy fixedpoint : " << current_acc_fixedpoint <<"\n";
std::cout << "current_bitwidth_conv : " << current_bitwidth_conv<<"\n";
std::cout << "current_bitwidth_fc : " << current_bitwidth_fc <<"\n";
current_error = (current_acc_original-current_acc_fixedpoint)/current_acc_original;
std::cout << "current_error : " << current_error <<"\n";

  if(current_error <= 0.01){
    current_bitwidth_conv -= 1;
  }

//loop over remaining batches to search on the correct 5000 images.

for (int k = 0 ; k<num_remain_batches; k++ ){
  ForwardFromTo(0, INPUT_LAYER+1);
}


} //end of while error < X
current_bitwidth_conv += 1;







}//end of SearchWeightFixedpoint batches




template <typename Dtype>
void Net<Dtype>::TestFixedpointSimple() {


  //size-3 = without top-5 layer
  //size-4 for imagenet's networks
  const int OUTPUT_LAYER = layers_.size() - 4;//  right before softmax. change to custom layer
//  const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer
  const int INPUT_LAYER = 2;
//  const int INPUT_LAYER = 0; //cifar // dont use this

  std::vector<float> mac_count;
  std::vector<float> weight_count;
  std::vector<float> input_count;
  std::vector<int> layer_index;
  int analyzing_layers = 0;
  for (int i=0; i<=OUTPUT_LAYER; i++){
    if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
    {
        // bottom_vecs_[i][0] input
        //layers_[i]->blobs()[0] weight
        // top_vecs_[i][0] output
      analyzing_layers++;
      weight_count.push_back(layers_[i]->blobs()[0]->count());
      input_count.push_back(bottom_vecs_[i][0]->count());
      if(strcmp (layers_[i]->type(),"InnerProduct") == 0 ){
          mac_count.push_back(layers_[i]->blobs()[0]->count()); //FC layers mac_count = weight_count
      }
      else{ //convolutional layer, shape size must be 4
          //top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3) * sizeof(weight)
          mac_count.push_back(top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3)*layers_[i]->blobs()[0]->count());
      }
      layer_index.push_back(i);
    }

  }

  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<layer_names_[layer_index[i]] <<"mac count: "<<mac_count[i]<< "\nweight count: "<<weight_count[i]<<"\n";

//getting optimized result
std::vector<int> input_bitwidth;
int weight_bitwidth;
int sigma_vector_count = 0;

vector<string> sigma_raw;
string temp_line;
//file format : weight bitwidth, + intput bitwidths
 std::ifstream solution("debugging/bitwidth_solution.txt");
 std::getline(solution,temp_line);
 boost::split(sigma_raw,temp_line,boost::is_any_of(","));
 if((sigma_raw.size() != analyzing_layers+1) ){
    std::cout<<" file size "<< sigma_raw.size()<<"\n";
    std::cout<<"wrong solution file, expecting a file with "<< analyzing_layers +1 << " numbers on the first line\n";
    return;
  }
 weight_bitwidth   = boost::lexical_cast<int> (sigma_raw[0]);
 std::cout<<sigma_raw[0]<<"\n";
 for (int i =1; i<sigma_raw.size(); i++){
   std::cout<<sigma_raw[i]<<"\n";
   input_bitwidth.push_back(boost::lexical_cast<int> (sigma_raw[i]));
 }



  std::cout<<"\n input count \n";
  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<input_count[i]<<", ";
  std::cout<<"\n\n";
  std::cout<<"\n weight bitwidth \n";
    std::cout<<weight_bitwidth<<", ";
  std::cout<<"\n input bitwidth \n";
  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<input_bitwidth[i]<<", ";
  std::cout<<"\n\n";


float original_acc = 0.689; //googlenet

std::cout<<"\n ----- doing forward with error -----\n";
std::cout<<"output layer name "<<layer_names_[OUTPUT_LAYER]<<"\n";

//int num_batches = 1; //cifar
//int num_batches = 500; //vgg
int num_batches = 250 ; //googlenet
//resnet50 batchsize 50
//resnet 152 batchsize 25
//int num_batches = 500; //resnet
//int num_batches = 250; //squeezenet mobilenet
//const int accuracy_index = 1; //googlenet
const int accuracy_index = 0; //vgg_16-19, cifar, etc
//0.7267
 AnalyzeFromToForward(0, layers_.size() - 1);


int max_analyzing_layer = analyzing_layers;

//ForwardFromTo(0, layers_.size() - 1);
//ForwardFromTo(0, layers_.size() - 1);

//AnalyzeFromToForwardWithError(INPUT_LAYER, layers_.size() - 1, layer_index[0],  sigma_result[0] );

//  AnalyzeFromToForwardFixedpoint(INPUT_LAYER, layers_.size() - 1, layer_index[2], 5,-2, 5,NULL);

//ForwardFromTo(INPUT_LAYER, layers_.size() - 1);
int current_bitwidth_conv = 16;
int current_bitwidth_fc = 16;

float current_error = 0; //relative error

float loss = 0;
Dtype current_acc_simulate = 0;
Dtype current_acc_fixedpoint = 0;
Dtype current_acc_original = 0;
std::vector<float> accuracy_simulate;
std::vector<float> accuracy_fixedpoint;
std::vector<float> accuracy_original;

for (int k = 0 ; k<num_batches; k++){
  //const vector<Blob<Dtype>*>& result =  Forward(&iter_loss);
  //AnalyzeFromToForwardWithError(0, layers_.size() - 1, OUTPUT_LAYER, upper_bound);
  //if (k!=0 ) // do this first to fetch data & correct the labels
    ForwardFromTo(0, layers_.size() - 1);

    { // the same dirty trick to reuse var name
      const vector<Blob<Dtype>*>& result  = net_output_blobs_;
      current_acc_original += result[accuracy_index]->cpu_data()[0];
      accuracy_original.push_back(result[accuracy_index]->cpu_data()[0]);
        std::cout << "Batch " << k << " " << result[accuracy_index]->cpu_data()[0] <<"\n";
    //  for (int j =0;j <result.size(); j++)
    //    std::cout << "Batch " << k << " original " << result[j]->cpu_data()[0] <<"\n";
    }

// doing the same thing with fixedpoint format

  for (int i=0; i<max_analyzing_layer; i++){

    //std::pair<int, int*> bitwidth_result = GetBitwidthFromError(sigma_result[i], NULL,layers_[layer_index[i]]->blobs()[0]->count(), true, 0.0);
    //AnalyzeFromToForwardFixedpoint(int start, int end, int layer_error, int a_int,int a_frac, int weight_int,int* weight_frac)
          if(i ==0 ){
            AnalyzeFromToForwardFixedpoint(INPUT_LAYER, layer_index[i], layer_index[i], 5,input_bitwidth[i], 5,weight_bitwidth);
          }
          else if (i == max_analyzing_layer -1){
            AnalyzeFromToForwardFixedpoint(layer_index[i-1]+1, layers_.size() - 1, layer_index[i], 5,input_bitwidth[i], 5,weight_bitwidth);
          }
        else{
            AnalyzeFromToForwardFixedpoint(layer_index[i-1]+1, layer_index[i], layer_index[i], 5,input_bitwidth[i], 5,weight_bitwidth);
          }
    //    bitwidth_result.push_back(GetBitwidthFromError(sigma_result[i], NULL,layers_[layer_index[i]]->blobs()[0]->count(), true, 0.0));

  }

  { // the same dirty trick to reuse var name
    const vector<Blob<Dtype>*>& result  = net_output_blobs_;
    current_acc_fixedpoint += result[accuracy_index]->cpu_data()[0];
    std::cout << "Batch Fixedpoint" << k << " " << result[accuracy_index]->cpu_data()[0] <<"\n";


    accuracy_fixedpoint.push_back(result[accuracy_index]->cpu_data()[0]);

//    for (int j =0;j <result.size(); j++)
//      std::cout << "Batch " << k << " fixedpoint " << result[j]->cpu_data()[0] <<"\n";

  }
 //comment fixedpoint section for testing with cifar

}//end of the loop over batches

//current_acc_simulate /= num_batches;
current_acc_fixedpoint /= num_batches;
current_acc_original /= num_batches;
std::cout << "accuracy original : " << current_acc_original<<"\n";
std::cout << "accuracy fixedpoint : " << current_acc_fixedpoint <<"\n";
//std::cout << "current_bitwidth_conv : " << current_bitwidth_conv<<"\n";
//std::cout << "current_bitwidth_fc : " << current_bitwidth_fc <<"\n";
current_error = (current_acc_original-current_acc_fixedpoint)/current_acc_original;
std::cout << "current_error : " << current_error <<"\n";






}//end of testFixedpointSimple



//test floating point rounding

template <typename Dtype>
void Net<Dtype>::TestFloatingpoint() {


  //size-3 = without top-5 layer
  //size-4 for imagenet's networks
  const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer
//  const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer
//  const int INPUT_LAYER = 2;
  const int INPUT_LAYER = 0; //cifar

  std::vector<float> mac_count;
  std::vector<float> weight_count;
  std::vector<float> input_count;
  std::vector<int> layer_index;
  int analyzing_layers = 0;
  for (int i=0; i<=OUTPUT_LAYER; i++){
    if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
    {
        // bottom_vecs_[i][0] input
        //layers_[i]->blobs()[0] weight
        // top_vecs_[i][0] output
      analyzing_layers++;
      weight_count.push_back(layers_[i]->blobs()[0]->count());
      input_count.push_back(bottom_vecs_[i][0]->count());
      if(strcmp (layers_[i]->type(),"InnerProduct") == 0 ){
          mac_count.push_back(layers_[i]->blobs()[0]->count()); //FC layers mac_count = weight_count
      }
      else{ //convolutional layer, shape size must be 4
          //top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3) * sizeof(weight)
          mac_count.push_back(top_vecs_[i][0]->shape(2)* top_vecs_[i][0]->shape(3)*layers_[i]->blobs()[0]->count());
      }
      layer_index.push_back(i);
    }

  }

  for(int i = 0; i <analyzing_layers; i++)
    std::cout<<layer_names_[layer_index[i]] <<"mac count: "<<mac_count[i]<< "\nweight count: "<<weight_count[i]<<"\n";

//getting optimized result
std::vector<float> sigma_result;

int sigma_vector_count = 0;

vector<string> sigma_raw;
string temp_line;


float original_acc = 0.689; //googlenet
std::cout<<"\n ----- doing forward with error -----\n";
std::cout<<"output layer name "<<layer_names_[OUTPUT_LAYER]<<"\n";

int num_batches = 1; //cifar
//int num_batches = 500; //vgg
//int num_batches = 10; //googlenet
//int num_batches = 200; //resnet
//const int accuracy_index = 1; //googlenet
const int accuracy_index = 0; //vgg_16-19
//0.7267
 AnalyzeFromToForward(0, layers_.size() - 1);

int max_analyzing_layer = analyzing_layers;

//ForwardFromTo(0, layers_.size() - 1);
//ForwardFromTo(0, layers_.size() - 1);
{
  const Dtype* out_array_simulated = top_vecs_[OUTPUT_LAYER][0]->cpu_data();
  std::ofstream myfile;
  myfile.open ("/home/minh/github/caffe/debugging/out_ref.txt");
  for(int j =0; j < top_vecs_[OUTPUT_LAYER][0]->count(); j ++ )
    myfile << out_array_simulated[j]<<",";
  myfile.close();
}

float loss = 0;
Dtype current_acc_simulate = 0;
Dtype current_acc_fixedpoint = 0;
Dtype current_acc_original = 0;
std::vector<float> accuracy_simulate;
std::vector<float> accuracy_fixedpoint;
std::vector<float> accuracy_original;

for (int k = 0 ; k<num_batches; k++){
  //const vector<Blob<Dtype>*>& result =  Forward(&iter_loss);
  //AnalyzeFromToForwardWithError(0, layers_.size() - 1, OUTPUT_LAYER, upper_bound);
  if (k!=0 ) // do this first to fetch data & correct the labels
    ForwardFromTo(0, layers_.size() - 1);

    { // the same dirty trick to reuse var name
      const vector<Blob<Dtype>*>& result  = net_output_blobs_;
      current_acc_original += result[accuracy_index]->cpu_data()[0];
      accuracy_original.push_back(result[accuracy_index]->cpu_data()[0]);

      for (int j =0;j <result.size(); j++)
        std::cout << "Batch " << k << " original " << result[j]->cpu_data()[0] <<"\n";
    }


// doing the same thing with floatingpoint format

  for (int i=0; i<max_analyzing_layer; i++){
          if(i ==0 ){
              AnalyzeFromToForwardFloatingpoint(INPUT_LAYER, layer_index[i], layer_index[i], 8,10);
          }
          else if (i == max_analyzing_layer -1){
            AnalyzeFromToForwardFloatingpoint(layer_index[i-1]+1, layers_.size() - 1, layer_index[i],8,10);
          }
        else{
            AnalyzeFromToForwardFloatingpoint(layer_index[i-1]+1, layer_index[i], layer_index[i], 8,10);
        }
  }

  {
    const Dtype* out_array_simulated = top_vecs_[OUTPUT_LAYER][0]->cpu_data();
    std::ofstream myfile;
    myfile.open ("/home/minh/github/caffe/debugging/out_floatingpoint.txt");
    for(int j =0; j < top_vecs_[OUTPUT_LAYER][0]->count(); j ++ )
      myfile << out_array_simulated[j]<<",";
    myfile.close();
  }

  { // the same dirty trick to reuse var name
    const vector<Blob<Dtype>*>& result  = net_output_blobs_;
    current_acc_fixedpoint += result[accuracy_index]->cpu_data()[0];
    std::cout << "Batch " << k << " " << result[accuracy_index]->cpu_data()[0] <<"\n";


    accuracy_fixedpoint.push_back(result[accuracy_index]->cpu_data()[0]);

    for (int j =0;j <result.size(); j++)
      std::cout << "Batch " << k << " floatingpoint " << result[j]->cpu_data()[0] <<"\n";
  }
 //comment fixedpoint section for testing with cifar

}//end of the loop over batches

current_acc_simulate /= num_batches;
current_acc_fixedpoint /= num_batches;
current_acc_original /= num_batches;
std::cout << "accuracy original : " << current_acc_original<<"\n";
for (int t = 0; t<accuracy_original.size();t++)
  std::cout<<"batch "<<t<<"  "<<accuracy_original[t]<<"\n";

std::cout << "accuracy fixedpoint : " << current_acc_fixedpoint <<"\n";
for (int t = 0; t<accuracy_fixedpoint.size();t++)
  std::cout<<"batch "<<t<<"  "<<accuracy_fixedpoint[t]<<"\n";


/*
  {
    const Dtype* out_array_simulated = top_vecs_[OUTPUT_LAYER][0]->cpu_data();
    std::ofstream myfile;
    myfile.open ("/home/minh/github/caffe/debugging/out_simulated.txt");
    for(int j =0; j < top_vecs_[OUTPUT_LAYER][0]->count(); j ++ )
      myfile << out_array_simulated[j]<<",";
    myfile.close();
  }
*/

}
//end of testfloatingpoint




template <typename Dtype>
void Net<Dtype>::AnalyzeInteger() {
  //new accuracy drop in number e.g.  0.05 = 5 percent loss

  std::cout <<"\n ------------- blobnames before ------- \n";
  vector<string> names = blob_names();
  for(std::vector<int>::size_type i = 0; i != names.size(); i++) {
    std::cout <<names[i] <<" ";
  }



//size-3 = without top-5 layer
//size-4 for imagenet's networks
const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer

std::cout <<"\n ------------- Layersize "<<layers_.size()<<"\n";
vector<string> layernames = layer_names();
for(std::vector<int>::size_type i = 0; i != layernames.size(); i++) {
  std::cout << i << " : "<<layernames[i] <<" \t";
}
std::cout <<"\n";

std::vector<float> input_count;
std::vector<int> layer_index;
int analyzing_layers = 0;
for (int i=0; i<= OUTPUT_LAYER; i++){
  if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
  {
    analyzing_layers++;
    input_count.push_back(bottom_vecs_[i][0]->count());
    layer_index.push_back(i);
  }

}

for(int i = 0; i <analyzing_layers; i++)
  std::cout<<input_count[i]<<", ";
std::cout<<"\n\n";
//for each batch, do the same
std::vector<float> max_abs;
std::vector<float> weight_max_abs;
for (int i =0; i<analyzing_layers ; i++){
    max_abs.push_back(0);
    weight_max_abs.push_back(0);
  }

int num_batches = 100;
  for (int k =0; k<num_batches; k ++)
  {
    AnalyzeFromToForward(0, layers_.size() - 1); //from 0, do once only
    for (int i =0; i<analyzing_layers ; i++)
    {
      float current_max = 0;
      const Dtype* input_array =  bottom_vecs_[layer_index[i]][0]->cpu_data();
      int input_count = bottom_vecs_[layer_index[i]][0]->count();
      for(int j = 0; j <input_count; j ++){
        if (fabs(input_array[j]) > current_max)
          current_max = fabs(input_array[j]);
      }
    //  std::cout<<"analyzing layer "<< layer_index[i]<<" max_abs " << current_max <<"\n";
        if(max_abs[i]< current_max)
        max_abs[i]=current_max;
    }


    std::cout<<"\nActivation max_ABS \n";
    for(int i = 0; i <analyzing_layers; i++)
      std::cout<<max_abs[i]<<", ";


  }


  for (int i =0; i<analyzing_layers ; i++)
  {
    const Dtype* weights  = layers_[layer_index[i]]->blobs()[0]->cpu_data();
    int weight_count = layers_[layer_index[i]]->blobs()[0]->count();

    float current_max = 0;
    for(int j = 0; j <weight_count; j ++){
      if (fabs(weights[j]) > current_max)
        current_max = fabs(weights[j]);
    }
  //  std::cout<<"analyzing layer "<< layer_index[i]<<" max_abs " << current_max <<"\n";

    weight_max_abs[i]=current_max;


  }
//weight
    std::cout<<"\nWeight max_ABS \n";
    for(int i = 0; i <analyzing_layers; i++)
      std::cout<<weight_max_abs[i]<<", ";

}//end of analyze integerbitwidth



template <typename Dtype>
Dtype Net<Dtype>::AnalyzeFromToForward(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    //if (debug_info_) { ForwardDebugInfo(i); }

    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }

  return loss;
}

//this code is copied from the answer https://stackoverflow.com/questions/5083465/fast-efficient-least-squares-fit-algorithm-in-c
// input the array of points, output m and b so that y = mx + b fits the data
template <typename Dtype> int Net<Dtype>::linear_regression(int n, const double x[], const double y[], double* m, double* b){
    double   sumx = 0.0;                      /* sum of x     */
    double   sumx2 = 0.0;                     /* sum of x**2  */
    double   sumxy = 0.0;                     /* sum of x * y */
    double   sumy = 0.0;                      /* sum of y     */
    double   sumy2 = 0.0;                     /* sum of y**2  */

    for (int i=0;i<n;i++){
        sumx  += x[i];
        sumx2 += x[i]*x[i];
        sumxy += x[i] * y[i];
        sumy  += y[i];
        sumy2 += y[i]*y[i];
    }

    double denom = (n * sumx2 - (sumx*sumx));
    if (denom == 0) {
        // singular matrix. can't solve the problem.
        *m = 0;
        *b = 0;
        return 1;
    }
    *m = (n * sumxy  -  sumx * sumy) / denom;
    *b = (sumy * sumx2  -  sumx * sumxy) / denom;
    if(*b < 0){
      std::cout<<"special case, approximate with y = mx";
      *m  = sumy/sumx;
      *b = 0;
    }
    return 0;
}


template <typename Dtype>
std::pair<int, int*> Net<Dtype>::GetBitwidthFromError(float std, Dtype* weight_error, int N_weights, bool same_weight, Dtype weight_std) {
//First convert from standard deviation to the boundary of uniform distribution
//Var = (b-a)^2/12 = (2*delta)^2/12 => sigma = 2*delta/sqrt(12) => delta = sigma*sqrt(12)/2 = 1.732*sigma
  //std::cout <<"GetBitwidthFromError "<<"N_weights\n";
  int a_frac ;
  float delta_std = std;//*1.732;

  if (delta_std > 1){
  //  std::cout<<" neg fraction bit " <<"\n";
    a_frac =  0 - int(fabs(log2 (delta_std )));
  }
  else if (delta_std <=0){
    a_frac = 16;
  }
  else{
#ifdef round_to_zero
    a_frac = int(fabs(log2 (delta_std ))) +1; //round up fraction bits in case round_to_zero
#endif
#ifdef round_to_nearest
    a_frac = int(fabs(log2 (delta_std ))); //round down fraction bits in case round_to_nearest
#endif
  }
  int* w_frac = new int[N_weights]();
  for (int i =0; i <N_weights; i++){
if(same_weight == false){

#ifdef round_to_zero
      w_frac[i] = int(fabs(log2 (weight_error[i] ))) + 1;
#endif
#ifdef round_to_nearest
      w_frac[i] = int(fabs(log2 (weight_error[i] ))) + 1;
#endif
}
  else{
    if (weight_std <=0){
      w_frac[i] = 16;
    }else{
      #ifdef round_to_zero
            w_frac[i] = int(fabs(log2 (weight_std ))) + 1;
      #endif
      #ifdef round_to_nearest
            w_frac[i] = int(fabs(log2 (weight_std ))) + 1;
      #endif
    }

  }

  }

  //std::cout<<" delta "<<delta_std << " frac "<< a_frac<<"\n";
  return std::make_pair(a_frac,w_frac);

}


template <typename Dtype>
Dtype Net<Dtype>::CalculateSTD( const Dtype* input_array, int N) {
//given an array, calculate its standard deviation, sample size is N
//calculate mean
  double sum = 0.0;
  for (int i=0; i< N ; i++){
    sum+= input_array[i];
  }
  double mean = sum/N;
  double sum_sqr = 0;
  for (int i=0; i< N ; i++){
    sum_sqr+= (input_array[i] - mean)*(input_array[i] - mean);
  }
  return sqrt(sum_sqr/(N-1));
}

template <typename Dtype>
Dtype Net<Dtype>::AnalyzeFromToForwardWithError(int start, int end, int layer_error, float delta_std, bool uniform, Dtype* weight_error) {
//note in branch experimental,  delta_std now is the standard deviation of the normal distribution,
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());

  size_t seed = 42;
  boost::random::mt19937 engine(seed);

  boost::function<double()> randn =
  boost::bind(boost::random::normal_distribution<>(0, delta_std), engine);

  const vector<string> layer_names =  Net<Dtype>::layer_names();

  Dtype loss = 0;
  // i == end -2 means last layers before accuracy & loss / i == 7, end == 9

  Dtype* temp_inputs = NULL;

  if (delta_std > 0 ){
//      std::cout<<"saving input to inject error "<<delta_std<<"\n";
      const Dtype* inputs  = bottom_vecs_[layer_error][0]->cpu_data();
      int inputs_count = bottom_vecs_[layer_error][0]->count();
      temp_inputs = new Dtype[inputs_count]();
      caffe_copy(inputs_count, inputs, temp_inputs);
  }

  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }

    //std::cout<<"Forward on layer "<< i <<" name "<<layer_names[i] <<"\n";
    //the split layers, inject error on bottom blobs before calling Forward
    //std::cout<<"top_vecs_ size "<<top_vecs_[i].size() <<"\n"; //end-2
    if (i== layer_error){ //inject error here
#ifndef GPU_ERROR_INJECTION
        Dtype* input_array =  bottom_vecs_[i][0]->mutable_cpu_data();

        Dtype* input_noise = new Dtype[bottom_vecs_[i][0]->count()]();
        //gaussian 99% => sigma = delta/3
        if(delta_std>0.0){
          //caffe_rng_gaussian<Dtype>(bottom_vecs_[i][0]->count(), 0.0, delta_std/3, input_noise);
          if(uniform == true){
            caffe_rng_uniform<Dtype>(bottom_vecs_[i][0]->count(), -delta_std, delta_std, input_noise);
          }
          else{
            caffe_rng_gaussian<Dtype>(bottom_vecs_[i][0]->count(), 0.0, delta_std, input_noise);
          }
        }
          //caffe_rng_uniform<Dtype>(bottom_vecs_[i][0]->count(), -delta_std*1.73, delta_std*1.73, input_noise);//test uniform
        for (int j = 0;j < bottom_vecs_[i][0]->count();j++){
          //input_array[j] = input_array[j] + randn();
          if(input_array[j] != 0.0) //inject on non zero value only
            input_array[j] = input_array[j] + input_noise[j];

        }
        delete input_noise;
#else //GPU error injection, faster but experimental
        Dtype* input_array =  bottom_vecs_[i][0]->mutable_gpu_data();

        Dtype* input_noise;
        CUDA_CHECK(cudaMallocHost(&input_noise, bottom_vecs_[i][0]->count() * sizeof(Dtype)));

        //gaussian 99% => sigma = delta/3
        if(delta_std>0.0){
          //caffe_rng_gaussian<Dtype>(bottom_vecs_[i][0]->count(), 0.0, delta_std/3, input_noise);
          if(uniform == true){
            caffe_gpu_rng_uniform<Dtype>(bottom_vecs_[i][0]->count(), -delta_std, delta_std, input_noise);
          }
          else{
            caffe_gpu_rng_gaussian<Dtype>(bottom_vecs_[i][0]->count(), 0.0, delta_std, input_noise);
          }
        }
        caffe_gpu_add_if<Dtype>(bottom_vecs_[i][0]->count(), input_array, input_noise);

        cudaFreeHost(input_noise);
/*        for (int j = 0;j < bottom_vecs_[i][0]->count();j++){
          //input_array[j] = input_array[j] + randn();
          if(input_array[j] != 0.0) //inject on non zero value only
            input_array[j] = input_array[j] + input_noise[j];
*/

#endif


//        std::cout <<"injecting errror "<< uniform <<" "<< delta_std<<"\n";
        //inject error to weight:
        if (weight_error != NULL){
          Dtype* weights  = layers_[i]->blobs()[0]->mutable_cpu_data();
          int weight_count = layers_[i]->blobs()[0]->count();
          std::cout<<"getting weigts to inject error : count "<<weight_count<<"\n";

          int count_nan = 0;
          for (int j = 0 ; j<weight_count ;j++){
              if (isnan(weight_error[j]) || weight_error[j] < 1e-8){
                count_nan ++;
                continue;
              }

              Dtype* rng_data = new Dtype[1];
              caffe_rng_uniform<Dtype>(1, -weight_error[j], weight_error[j], rng_data);
              weights[j] = weights[j]+rng_data[0];

          }

          std::cout <<"count nan "<< count_nan <<"\n";
        //  std::cout <<"output some to test weigts "<< weights[0]<<" "<< weights[100]<<" "<<weights[200]<<"\n";
        }

    }

    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);



    loss += layer_loss;
    //if (debug_info_) { ForwardDebugInfo(i); }

    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }

  if (delta_std > 0 &&  temp_inputs!=NULL ){
//      std::cout<<"return original input values"<<delta_std<<"\n";
      Dtype* inputs  = bottom_vecs_[layer_error][0]->mutable_cpu_data();
      int inputs_count = bottom_vecs_[layer_error][0]->count();
      caffe_copy(inputs_count, temp_inputs, inputs);
      delete temp_inputs;
  }
    {
//      const Dtype* result_vec_error = net_output_blobs_[0]->cpu_data();
//      float current_acc = result_vec_error[0];
//      std::cout<<"current acc " <<current_acc<<"\n";
    }

  return loss;
}

template <typename Dtype>
void Net<Dtype>::ProcessLambdaThetaEqual(double delta_std)
{
  //write sigma file
  //size-3 = without top-5 layer
  //size-4 for imagenet's networks
  const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer
  //const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer
  const int INPUT_LAYER = 2; //cifar

  std::vector<int> layer_index;
  int analyzing_layers = 0;
  for (int i=0; i<=OUTPUT_LAYER; i++){
    if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
    {
      analyzing_layers++;
      layer_index.push_back(i);
    }
  }

//get theta_sigma
std::vector<float> lambda_result;
std::vector<float> theta_result;
int lambda_vector_count = 0;

vector<string> lambda_raw;
string temp_line;

std::cout<<"\n lambda \n";
 std::ifstream solution("debugging/lambda_theta.txt");
 std::getline(solution,temp_line);
 boost::split(lambda_raw,temp_line,boost::is_any_of(","));
 for (int i =0; i<lambda_raw.size(); i++){
   std::cout<<lambda_raw[i]<<",";
   lambda_result.push_back(boost::lexical_cast<float> (lambda_raw[i]));
 }
//get 2nd line contains theta
std::cout<<"\n theta \n";
 std::getline(solution,temp_line);
 boost::split(lambda_raw,temp_line,boost::is_any_of(","));
 for (int i =0; i<lambda_raw.size(); i++){
   std::cout<<lambda_raw[i]<<"\n";
   theta_result.push_back(boost::lexical_cast<float> (lambda_raw[i]));
 }

 lambda_vector_count = theta_result.size();

 //getting equally distributed sigma
  std::vector<float> sigma_result;
  double coeff_mult = 1.0/lambda_vector_count;
  for (int i = 0; i< lambda_vector_count ;i++){
      sigma_result.push_back(sqrt(coeff_mult)*delta_std*lambda_result[i] + theta_result[i]);
  }
  if(lambda_vector_count< analyzing_layers)
  for (int i = 0; i< analyzing_layers - lambda_vector_count ;i++)
  sigma_result.push_back(0.0);

  std::cout<<"sigma result\n";
  for(int i =0; i<  lambda_vector_count; i++)
    std::cout<<sigma_result[i]<<",";

  std::cout<<"\n";

  {
    std::cout<<"writing back to sigma_solution.txt\n";
    std::ofstream myfile;
    myfile.open ("debugging/sigma_solution.txt");

    for(int j =0; j <analyzing_layers ; j++){
          if (j == analyzing_layers-1)
              myfile << sigma_result[j];
          else
              myfile << sigma_result[j]<<",";
      }
    myfile.close();
  }

}

template <typename Dtype>
Dtype Net<Dtype>::AnalyzeFromToForwardWithErrorEqualDistributed(int start, int end, float delta_std) {

  //write sigma file
  //size-3 = without top-5 layer
  //size-4 for imagenet's networks
  const int OUTPUT_LAYER = layers_.size() - 4;// size-3 = right before softmax. change to custom layer
  //const int OUTPUT_LAYER = layers_.size() - 3;// size-3 = right before softmax. change to custom layer
  const int INPUT_LAYER = 2; //cifar

  std::vector<int> layer_index;
  int analyzing_layers = 0;
  for (int i=0; i<=OUTPUT_LAYER; i++){
    if (strcmp (layers_[i]->type(),"Convolution") == 0 || strcmp (layers_[i]->type(),"InnerProduct") == 0)
    {
      analyzing_layers++;
      layer_index.push_back(i);
    }
  }

  //getting optimized result
  std::vector<float> sigma_result;

  int sigma_vector_count = 0;

  vector<string> sigma_raw;
  string temp_line;

   std::ifstream solution("debugging/sigma_solution.txt");
   std::getline(solution,temp_line);
   boost::split(sigma_raw,temp_line,boost::is_any_of(","));
   if((sigma_raw.size() != analyzing_layers) ){
      std::cout<<"wrong solution file, expecting a file with "<< analyzing_layers << " numbers on the first line\n";
    }
   for (int i =0; i<sigma_raw.size(); i++){
    // std::cout<<sigma_raw[i]<<"\n";
     sigma_result.push_back(boost::lexical_cast<float> (sigma_raw[i]));
   }
/*
    std::cout<<"\n sigma result \n";
    for(int i = 0; i <analyzing_layers; i++)
      std::cout<<sigma_result[i]<<", ";
    std::cout<<"\n\n";
*/
  //printf(" to here\n");
  //done, calling testigma as usual
  ForwardFromTo(0, layers_.size() - 1);
  for (int i=0; i<analyzing_layers; i++){
          if(i ==0 ){
            //use this to simulate the behaviour of injecting error
            AnalyzeFromToForwardWithError(INPUT_LAYER, layer_index[i], layer_index[i], sigma_result[i],true );
          }
          else if (i == analyzing_layers -1){
            AnalyzeFromToForwardWithError(layer_index[i-1]+1, layers_.size() - 1, layer_index[i], sigma_result[i],true);
          }
        else{
            AnalyzeFromToForwardWithError(layer_index[i-1]+1, layer_index[i],layer_index[i], sigma_result[i], true);
        }

    }

  return 0.0;//for compatibility
}




template <typename Dtype>
Dtype Net<Dtype>::AnalyzeFromToForwardWithErrorSameWeight(int start, int end, int layer_error, float delta_std, Dtype weight_error, bool uniform) {
//note in branch experimental,  delta_std now is the standard deviation of the normal distribution,
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());

  //size_t seed = time(NULL);
  size_t seed = 42;
  boost::random::mt19937 engine(seed);
/*  boost::uniform_int<> one_to_two( 1, 2 );
  boost::variate_generator< RNGType, boost::uniform_int<> >
  dice(rng, one_to_two);
*/
  boost::function<double()> randn =
  boost::bind(boost::random::normal_distribution<>(0, delta_std), engine);

  /*boost::normal_distribution<> nd(0.0, delta_std); //std = 1.0

  boost::variate_generator<boost::mt19937&,
                           boost::normal_distribution<> > var_nor(rng, nd);
*/

  //std::cout<<"test dice " << randn() << randn()<<randn()<<randn() <<"\n";

  const vector<string> layer_names =  Net<Dtype>::layer_names();

  Dtype loss = 0;
  // i == end -2 means last layers before accuracy & loss / i == 7, end == 9
  Dtype* temp_weight = NULL;
  Dtype* temp_inputs = NULL;
  if (weight_error > 0 ){
      std::cout<<"saving weight to inject error "<<weight_error<<"\n";
      const Dtype* weights  = layers_[layer_error]->blobs()[0]->cpu_data();
      int weight_count = layers_[layer_error]->blobs()[0]->count();
      temp_weight = new Dtype[weight_count]();
      caffe_copy(weight_count, weights, temp_weight);
  }
  if (delta_std > 0 ){
      std::cout<<"saving input to inject error "<<delta_std<<"\n";
      const Dtype* inputs  = bottom_vecs_[layer_error][0]->cpu_data();
      int inputs_count = bottom_vecs_[layer_error][0]->count();
      temp_inputs = new Dtype[inputs_count]();
      caffe_copy(inputs_count, inputs, temp_inputs);
  }

  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }

  //std::cout<<"Forward on layer "<< i <<" name "<<layer_names[i] <<"\n";
    //the split layers, inject error on bottom blobs before calling Forward
    //std::cout<<"top_vecs_ size "<<top_vecs_[i].size() <<"\n"; //end-2
    if (i== layer_error){ //inject error here
//#ifdef DEBUG_CODE
        std::cout<<"injecting error "<< delta_std<<"\n";
        std::cout<<"vector size "<<bottom_vecs_[i].size()<<"\n";
        std::cout<<"blob shape "<<bottom_vecs_[i][0]->shape_string()<<"\n";
//#endif
        Dtype* input_array =  bottom_vecs_[i][0]->mutable_cpu_data();

        Dtype* input_noise = new Dtype[bottom_vecs_[i][0]->count()]();
        //gaussian 99% => sigma = delta/3
        if(delta_std>0.0){
          //caffe_rng_gaussian<Dtype>(bottom_vecs_[i][0]->count(), 0.0, delta_std/3, input_noise);
          if(uniform == true)
            caffe_rng_uniform<Dtype>(bottom_vecs_[i][0]->count(), 0.0, delta_std, input_noise);
          else
            caffe_rng_gaussian<Dtype>(bottom_vecs_[i][0]->count(), 0.0, delta_std, input_noise);
          }
          //caffe_rng_uniform<Dtype>(bottom_vecs_[i][0]->count(), -delta_std*1.73, delta_std*1.73, input_noise);//test uniform
        for (int j = 0;j < bottom_vecs_[i][0]->count();j++){
          //input_array[j] = input_array[j] + randn();
          if(input_array[j] != 0.0) //inject on non zero value only
            input_array[j] = input_array[j] + input_noise[j];
          /*
          if(dice()==1)
            input_array[j] = input_array[j]+delta;
          else
            input_array[j] = input_array[j]-delta;
            */
        }

        //inject error to weight:
        if (weight_error != 0){
          Dtype* weights  = layers_[i]->blobs()[0]->mutable_cpu_data();
          int weight_count = layers_[i]->blobs()[0]->count();
          std::cout<<"getting weigts to inject error : count "<<weight_count<<"\n";

          int count_less_than_delta = 0;
          for (int j = 0 ; j<weight_count ;j++){
      /*        if (isnan(weight_error[j]) || weight_error[j] < 1e-8){
                count_nan ++;
                continue;
              }
      */
              Dtype* rng_data = new Dtype[1];

              caffe_rng_uniform<Dtype>(1, -weight_error, weight_error, rng_data);
/*              weights[j] = weights[j]+rng_data[0];
*/
//testing May 7

                if(rng_data[0] >0 )
                  weights[j] = weights[j]+weight_error;
                else
                  weights[j] = weights[j]-weight_error;

              if(fabs(weights[j])<weight_error){
                  count_less_than_delta = count_less_than_delta+1;
              //    weights[j] = 0.0; //todo: change later with value name weight limits
              }
              //if(j%100==0)
                //std::cout << "Test rand "<<rng_data[0]<<"\n";

          }

          std::cout <<"count count_less_than_delta "<< count_less_than_delta <<"\n";
        //  std::cout <<"output some to test weigts "<< weights[0]<<" "<< weights[100]<<" "<<weights[200]<<"\n";
        //  std::cout <<"output some to test weigts "<< weights[10]<<" "<< weights[20]<<" "<<weights[30]<<"\n";
        }

    }

    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);



    loss += layer_loss;
    //if (debug_info_) { ForwardDebugInfo(i); }

    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }

    {
      const Dtype* result_vec_error = net_output_blobs_[0]->cpu_data();
      float current_acc = result_vec_error[0];
      std::cout<<"current acc " <<current_acc<<"\n";
    }

    if (weight_error > 0 &&  temp_weight!=NULL){
        std::cout<<"return original weight values "<<weight_error<<"\n";
        Dtype* weights  = layers_[layer_error]->blobs()[0]->mutable_cpu_data();
        int weight_count = layers_[layer_error]->blobs()[0]->count();
        caffe_copy(weight_count, temp_weight, weights);
        delete temp_weight;
    }
    if (delta_std > 0 &&  temp_inputs!=NULL ){
        std::cout<<"return original input values"<<delta_std<<"\n";
        Dtype* inputs  = bottom_vecs_[layer_error][0]->mutable_cpu_data();
        int inputs_count = bottom_vecs_[layer_error][0]->count();
        caffe_copy(inputs_count, temp_inputs, inputs);
        delete temp_inputs;
    }

  return loss;
}


//all weights use the same integer bitwidth.
template <typename Dtype>
Dtype Net<Dtype>::AnalyzeFromToForwardFixedpoint(int start, int end, int layer_error, int a_int,int a_frac, int weight_int,int weight_frac) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
//  std::cout <<"AnalyzeFromToForwardFixedpoint "<< start<< " "<<end << " "<< layer_error << " " <<a_frac<<" "<<weight_frac << "\n";
  int power_2 [] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288 };
  const vector<string> layer_names =  Net<Dtype>::layer_names();

  Dtype loss = 0;

  Dtype* temp_weight = NULL;
  if (weight_frac != -100 ){
  //    std::cout<<"saving weight to inject error "<<weight_error<<"\n";
      const Dtype* weights  = layers_[layer_error]->blobs()[0]->cpu_data();
      int weight_count = layers_[layer_error]->blobs()[0]->count();
      temp_weight = new Dtype[weight_count]();
      caffe_copy(weight_count, weights, temp_weight);
  }

  // i == end -2 means last layers before accuracy & loss / i == 7, end == 9
  //this must run as last phase, no need to save input ?
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }

    if (i== layer_error){ //inject error here

        Dtype* input_array =  bottom_vecs_[i][0]->mutable_cpu_data();

        for (int j = 0;j < bottom_vecs_[i][0]->count();j++){
//          if(input_array[j] != 0.0) //inject on non zero value only
//            input_array[j] = input_array[j] + input_noise[j];

            //  if(j%1000 == 0)
            //    std::cout<< " original "<< input_array[j] << " frac_bit "<<a_frac <<"\n";
              if(a_frac != -100){
              if(input_array[j] != 0.0){
#ifdef round_to_zero
                  if(a_frac > 0)
                    input_array[j] = double(int(input_array[j]*power_2[a_frac]))/double(power_2[a_frac]);
                  else
                    input_array[j] = double(int(input_array[j]/power_2[abs(a_frac)]))*double(power_2[abs(a_frac)]);
#endif
#ifdef round_to_nearest
                  if(a_frac>0)
                      input_array[j] = double(int(round(input_array[j]*power_2[a_frac])))/double(power_2[a_frac]);
                  else{
                      input_array[j] = double(int(round(input_array[j]/power_2[abs(a_frac)])))*double(power_2[abs(a_frac)]);

                  }
#endif
                }
              }
              //else{
                //   std::cout<< "\n\n  a_frac = invalid,no rounding "<< layer_error <<"\n\n";
            //  }

                //debugging
          //      if(j%1000 == 0)
          //        std::cout<< " converted "<< input_array[j] <<"\n";



        }
        //inject error to weight:
        if (weight_frac != -100){
          Dtype* weights  = layers_[i]->blobs()[0]->mutable_cpu_data();
          int weight_count = layers_[i]->blobs()[0]->count();

          for (int j = 0 ; j<weight_count ;j++){
    //         if(j%100 == 0)
    //             std::cout<< " original "<< weights[j] << " frac_bit "<<weight_frac[j] <<"\n";
            #ifdef round_to_zero
                              weights[j] = double(int(weights[j]*power_2[weight_frac[j]]))/double(power_2[weight_frac[j]]);
                              //intentional, compilation error if using round_to_zero, currently not supported;
            #endif
            #ifdef round_to_nearest
                              weights[j] = double(int(round(weights[j]*power_2[weight_frac])))/double(power_2[weight_frac]);
            #endif
            /*
            if(fabs(weights[j]) < weight_limit)
              weights[j] = 0;
            */
    //        if(j%100 == 0)
    //            std::cout<< " converted "<< weights[j] <<"\n";

          }


        }

    } //end of if(layer error)

    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);



    loss += layer_loss;
    //if (debug_info_) { ForwardDebugInfo(i); }

    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }// end of loop from start_layer to end_layer

    if (weight_frac!= -100 &&  temp_weight!=NULL){
      //  std::cout<<"return original weight values "<<weight_error<<"\n";
        Dtype* weights  = layers_[layer_error]->blobs()[0]->mutable_cpu_data();
        int weight_count = layers_[layer_error]->blobs()[0]->count();
        caffe_copy(weight_count, temp_weight, weights);
        delete temp_weight;
    }

  return loss;
}


//this function to round floating point data to arbitrary precision
//required bit manipulation ops
typedef enum {ERROR = -1, FALSE, TRUE} LOGICAL;
#define BOOL(x) (!(!(x)))
#define BitSet(arg,posn) ((arg) | (1L << (posn)))
#define BitClr(arg,posn) ((arg) & ~(1L << (posn)))
#define BitTst(arg,posn) BOOL((arg) & (1L << (posn)))
#define BitFlp(arg,posn) ((arg) ^ (1L << (posn)))
//clear all nBits from the end
#define BitsClr(arg,nBits) ((arg) & (~(unsigned long)((unsigned long)(1UL << (nBits+1)) -1UL)))

template <typename Dtype>
float Net<Dtype>:: round_float (float data, int precision){
    //too much warning in this func. need to fix
    //precision = mantissa bit
    if (precision > 23) return 0.0; //arth error
    int bits_to_clear = 23 - precision; //clear from the end to round to zero;
    int position_to_set = 23 - precision;
    int bit_to_test = bits_to_clear-1;//9+precision; //bit to test for rounding. the next bit after precision
    //cout<<"bit_toclear "<<bit_to_test<<endl;
    bool round_up = BitTst(*((int*)&data),bit_to_test);
//0 10000001 0110100011 1101110001100
//0 10000001 0110100011 0000000000000
//0 10000001 0000000000 0000000000000
//0 10000001 0000000001 0000000000000
    float result = 0;
    int dataint_cleared = (BitsClr(*((int*)&data),bits_to_clear-1));

    //std::cout<< "round up "<< round_up<<endl;
    //cout << "data "<<binary_text(*(int*)&data) << endl;
    //cout << "dataint cleared "<<binary_text(dataint_cleared) << endl;

    if (!round_up){
      result = *(float*)&dataint_cleared;
    }
    else{
      float data_cleared = *(float*)&dataint_cleared;

      int dataint_expsign = (BitsClr(*((int*)&data),32-9-1)); //clear all bits except sign + exponent;

      int dataint_ulp = BitSet(dataint_expsign,position_to_set);//set 1 ulp bit from the above data
      //cout << binary_text(dataint_expsign) << endl;
      //cout << binary_text(dataint_ulp) << endl;
      result = *(float*)&dataint_cleared + *(float*)&dataint_ulp  -  *(float*)&dataint_expsign;


    }
  //  cout << binary_text(data) << endl;
  return result;
}


//this func used for floating point rounding with mantissa bitwidth M
//all weights use the same mantissa bitwidth.
template <typename Dtype>
Dtype Net<Dtype>::AnalyzeFromToForwardFloatingpoint(int start, int end, int layer_error ,int a_frac,int weight_frac) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  std::cout <<"AnalyzeFromToForwardFloatingpoint "<< start<< " "<<end << " "<< layer_error << " " <<a_frac << "\n";
  const vector<string> layer_names =  Net<Dtype>::layer_names();

  Dtype loss = 0;
  // i == end -2 means last layers before accuracy & loss / i == 7, end == 9
  //this must run as last phase, no need to save input ?
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }

    if (i== layer_error){ //inject error here

        Dtype* input_array =  bottom_vecs_[i][0]->mutable_cpu_data();
        std::cout<<"weights_shape "<<layers_[i]->blobs()[0]->shape_string()<<"\n";
        for (int j = 0;j < bottom_vecs_[i][0]->count();j++){
//          if(input_array[j] != 0.0) //inject on non zero value only
//            input_array[j] = input_array[j] + input_noise[j];

          //    if(j%5000 == 0)
          //      std::cout<< " original "<< input_array[j] << " frac_bit "<<a_frac <<"\n";

              if(input_array[j] != 0.0){
                      input_array[j] = double(round_float((float)input_array[j],a_frac));

                }

          //      if(j%5000 == 0)
        //          std::cout<< " converted "<< input_array[j] <<"\n";

        }

        Dtype* weights  = layers_[i]->blobs()[0]->mutable_cpu_data();
        int weight_count = layers_[i]->blobs()[0]->count();

        //save bitwidth to file for checking
        {
          std::ofstream myfile;

          myfile.open (("debugging/weights"+layer_names[i]).c_str(),std::ofstream::app);
          for(int j =0; j < weight_count; j ++ ){
            myfile << weights[j]<<",";
            if (j != 0 && (j %24 == 0))
              myfile <<"\n";
            }
          myfile.close();
        }






    }

    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);



    loss += layer_loss;
    //if (debug_info_) { ForwardDebugInfo(i); }

    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }

    {
      const Dtype* result_vec_error = net_output_blobs_[0]->cpu_data();
      float current_acc = result_vec_error[0];
      std::cout<<"current acc " <<current_acc<<"\n";
    }

  return loss;
}




template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    for (int c = 0; c < before_backward_.size(); ++c) {
      before_backward_[c]->run(i);
    }
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
    for (int c = 0; c < after_backward_.size(); ++c) {
      after_backward_[c]->run(i);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (H5Fis_hdf5(trained_filename.c_str())) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
