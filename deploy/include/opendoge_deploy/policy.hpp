#pragma once

#include <array>
#include <memory>
#include <string>

#include "opendoge_deploy/types.hpp"

namespace opendoge
{

/// Deployment contract read from the policy artifact. The main loop validates
/// this against DeployConfig before enabling control, so a model trained for a
/// different observation/action layout can never be run by accident.
struct PolicyInfo
{
  bool valid{false};
  std::size_t observation_dim{0};
  std::size_t history_length{0};
  std::size_t action_dim{0};
  double control_dt_s{0.0};
  std::array<double, kNumJoints> action_scale{};
  std::array<double, kNumJoints> default_joint_positions{};
  std::array<std::string, kNumJoints> joint_names{};
};

class Policy
{
public:
  virtual ~Policy() = default;
  virtual bool load(const std::string & path, std::string & error) = 0;
  virtual bool infer(
    const ObsHistory & history,
    std::array<double, kNumJoints> & action,
    std::string & error) = 0;
  virtual const PolicyInfo & info() const = 0;
};

std::unique_ptr<Policy> makePolicy(const std::string & backend);

}  // namespace opendoge
