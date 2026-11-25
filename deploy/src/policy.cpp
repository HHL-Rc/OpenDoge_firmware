#include "opendoge_deploy/policy.hpp"

#include <algorithm>
#include <cmath>

namespace opendoge
{
namespace
{

PolicyInfo defaultPolicyInfo()
{
  PolicyInfo info;
  info.valid = true;
  info.observation_dim = kObsDim;
  info.history_length = kObsHistoryLength;
  info.action_dim = kNumJoints;
  info.control_dt_s = kPolicyDt;
  info.action_scale = {
    0.25, 0.4, 0.4,  0.25, 0.4, 0.4,
    0.25, 0.4, 0.4,  0.25, 0.4, 0.4};
  info.default_joint_positions = {
    0.0, 0.8, -1.6,  0.0, -0.8, 1.6,
    0.0, 0.8, -1.6,  0.0, -0.8, 1.6};
  info.joint_names = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
  return info;
}

class NonePolicy final : public Policy
{
public:
  bool load(const std::string &, std::string &) override
  {
    info_ = defaultPolicyInfo();
    return true;
  }

  bool infer(
    const ObsHistory &,
    std::array<double, kNumJoints> & action,
    std::string &) override
  {
    action.fill(0.0);
    return true;
  }

  const PolicyInfo & info() const override { return info_; }

private:
  PolicyInfo info_{};
};

class MissingOnnxPolicy final : public Policy
{
public:
  bool load(const std::string &, std::string & error) override
  {
    error = "opendoge_deploy was built without ONNX Runtime";
    return false;
  }

  bool infer(
    const ObsHistory &,
    std::array<double, kNumJoints> &,
    std::string & error) override
  {
    error = "ONNX backend unavailable";
    return false;
  }

  const PolicyInfo & info() const override { return info_; }

private:
  PolicyInfo info_{};
};

}  // namespace

std::unique_ptr<Policy> makeOnnxPolicy();

std::unique_ptr<Policy> makePolicy(const std::string & backend)
{
  if (backend == "none") {
    return std::make_unique<NonePolicy>();
  }
  if (backend == "onnx") {
#ifdef OPENDOGE_HAS_ONNX
    return makeOnnxPolicy();
#else
    return std::make_unique<MissingOnnxPolicy>();
#endif
  }
  return nullptr;
}

}  // namespace opendoge
