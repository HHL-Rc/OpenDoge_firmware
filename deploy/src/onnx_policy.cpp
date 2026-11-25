#include "opendoge_deploy/policy.hpp"

#ifdef OPENDOGE_HAS_ONNX

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace opendoge
{
namespace
{
class OnnxPolicy final : public Policy
{
public:
  OnnxPolicy()
  : env_(ORT_LOGGING_LEVEL_WARNING, "opendoge_deploy"),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  }

  bool load(const std::string & path, std::string & error) override
  {
    try {
      session_ = std::make_unique<Ort::Session>(env_, path.c_str(), session_options_);
      auto input_name = session_->GetInputNameAllocated(0, allocator_);
      auto output_name = session_->GetOutputNameAllocated(0, allocator_);
      input_name_ = input_name.get();
      output_name_ = output_name.get();
      if (!validateIo(error) || !readMetadata(error)) {
        session_.reset();
        return false;
      }
    } catch (const std::exception & exc) {
      error = exc.what();
      return false;
    }
    return true;
  }

  bool infer(
    const ObsHistory & history,
    std::array<double, kNumJoints> & action,
    std::string & error) override
  {
    if (!session_) {
      error = "ONNX session is not loaded";
      return false;
    }

    std::array<float, kObsHistoryDim> input{};
    for (std::size_t i = 0; i < kObsHistoryDim; ++i) {
      if (!std::isfinite(history[i])) {
        error = "ONNX input contains NaN or Inf";
        return false;
      }
      input[i] = static_cast<float>(history[i]);
    }
    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(kObsHistoryDim)};
    auto tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input.data(), input.size(), input_shape.data(), input_shape.size());

    const char * input_names[] = {input_name_.c_str()};
    const char * output_names[] = {output_name_.c_str()};
    try {
      auto outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
      const auto * out = outputs[0].GetTensorData<float>();
      for (std::size_t i = 0; i < kNumJoints; ++i) {
        const double value = static_cast<double>(out[i]);
        if (!std::isfinite(value)) {
          error = "ONNX output contains NaN or Inf";
          return false;
        }
        // Policy output is intentionally unbounded (training applies no clip).
        action[i] = value;
      }
    } catch (const std::exception & exc) {
      error = exc.what();
      return false;
    }
    return true;
  }

  const PolicyInfo & info() const override { return info_; }

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  PolicyInfo info_{};

  static std::string shapeString(const std::vector<int64_t> & shape)
  {
    std::ostringstream ss;
    ss << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      ss << shape[i];
    }
    ss << "]";
    return ss.str();
  }

  static std::string trim(const std::string & input)
  {
    const auto first = std::find_if_not(input.begin(), input.end(), [](unsigned char c) {
      return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char c) {
      return std::isspace(c) != 0;
    }).base();
    return first >= last ? std::string{} : std::string(first, last);
  }

  static std::vector<double> parseNumberArray(const std::string & text)
  {
    std::vector<double> values;
    std::string normalized = text;
    for (char & c : normalized) {
      if (c == '[' || c == ']' || c == ',') {
        c = ' ';
      }
    }
    std::stringstream ss(normalized);
    double value = 0.0;
    while (ss >> value) {
      values.push_back(value);
    }
    return values;
  }

  static std::vector<std::string> parseStringArray(const std::string & text)
  {
    std::vector<std::string> values;
    std::size_t position = 0;
    while (true) {
      const auto open = text.find('"', position);
      if (open == std::string::npos) {
        break;
      }
      const auto close = text.find('"', open + 1);
      if (close == std::string::npos) {
        break;
      }
      values.push_back(text.substr(open + 1, close - open - 1));
      position = close + 1;
    }
    return values;
  }

  std::string metadata(const char * key)
  {
    auto value = session_->GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator_);
    return value ? std::string(value.get()) : std::string{};
  }

  bool readMetadata(std::string & error)
  {
    const auto observation_dim = metadata("observation_dim");
    const auto history_length = metadata("history_length");
    const auto control_dt = metadata("control_dt_seconds");
    const auto history_order = metadata("history_order");
    const auto action_scales = metadata("action_scales");
    const auto default_positions = metadata("default_joint_positions");
    const auto joint_names = metadata("joint_names");
    if (observation_dim.empty() || history_length.empty() || control_dt.empty() ||
        history_order.empty() || action_scales.empty() || default_positions.empty() ||
        joint_names.empty())
    {
      error = "ONNX metadata is missing the SuperMimic deployment contract";
      return false;
    }

    try {
      info_.observation_dim = static_cast<std::size_t>(std::stoul(observation_dim));
      info_.history_length = static_cast<std::size_t>(std::stoul(history_length));
      info_.control_dt_s = std::stod(control_dt);
    } catch (const std::exception & exc) {
      error = std::string("ONNX metadata parse failed: ") + exc.what();
      return false;
    }
    info_.action_dim = kNumJoints;
    if (info_.observation_dim != kObsDim || info_.history_length != kObsHistoryLength ||
        std::abs(info_.control_dt_s - kPolicyDt) > 1.0e-6)
    {
      error = "ONNX metadata dimensions/timing do not match this deploy build";
      return false;
    }
    std::string history_order_value = trim(history_order);
    if (history_order_value.size() >= 2 &&
        history_order_value.front() == '"' && history_order_value.back() == '"')
    {
      history_order_value =
        history_order_value.substr(1, history_order_value.size() - 2);
    }
    if (history_order_value != "frame_major_oldest_first") {
      error = "ONNX history_order must be frame_major_oldest_first, got " + history_order;
      return false;
    }

    const auto scales = parseNumberArray(action_scales);
    const auto defaults = parseNumberArray(default_positions);
    const auto names = parseStringArray(joint_names);
    if (scales.size() != kNumJoints || defaults.size() != kNumJoints ||
        names.size() != kNumJoints)
    {
      error = "ONNX metadata arrays must each contain 12 entries";
      return false;
    }
    for (std::size_t i = 0; i < kNumJoints; ++i) {
      info_.action_scale[i] = scales[i];
      info_.default_joint_positions[i] = defaults[i];
      info_.joint_names[i] = names[i];
    }
    info_.valid = true;
    return true;
  }

  bool validateIo(std::string & error)
  {
    const auto input_type_info = session_->GetInputTypeInfo(0);
    const auto output_type_info = session_->GetOutputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
    const auto input_type = input_info.GetElementType();
    const auto output_type = output_info.GetElementType();
    const auto input_shape = input_info.GetShape();
    const auto output_shape = output_info.GetShape();

    if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      error = "ONNX input must be float tensor";
      return false;
    }
    if (output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      error = "ONNX output must be float tensor";
      return false;
    }
    if (input_shape.size() != 2 ||
        input_shape[1] != static_cast<int64_t>(kObsHistoryDim))
    {
      error = "ONNX input shape must be [batch," + std::to_string(kObsHistoryDim) +
        "], got " + shapeString(input_shape);
      return false;
    }
    if (output_shape.size() != 2 ||
        output_shape[1] != static_cast<int64_t>(kNumJoints))
    {
      error = "ONNX output shape must be [batch,12], got " + shapeString(output_shape);
      return false;
    }
    return true;
  }
};
}  // namespace

std::unique_ptr<Policy> makeOnnxPolicy()
{
  return std::make_unique<OnnxPolicy>();
}

}  // namespace opendoge

#endif
