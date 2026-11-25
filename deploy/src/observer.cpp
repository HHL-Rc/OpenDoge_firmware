#include "opendoge_deploy/observer.hpp"

#include <algorithm>
#include <cmath>

namespace opendoge
{

std::array<double, 2> gaitPhaseFeatures(
  double elapsed_s,
  double control_dt,
  double period_s,
  double threshold,
  const OperatorCommand & command)
{
  const double dt = std::max(control_dt, 1.0e-6);
  const double period = std::max(period_s, 1.0e-6);
  const long period_steps = std::max(1L, std::lround(period / dt));
  const long steps = std::lround(std::max(0.0, elapsed_s) / dt);
  const double phase =
    static_cast<double>(steps % period_steps) / static_cast<double>(period_steps);
  const double magnitude =
    std::hypot(command.vx, command.vy) + std::abs(command.yaw_rate);
  if (magnitude <= threshold) {
    return {0.0, 0.0};
  }
  const double angle = 2.0 * 3.14159265358979323846 * phase;
  return {std::sin(angle), std::cos(angle)};
}

ObsFrame buildObservationFrame(
  const std::array<MotorState, kNumJoints> & states,
  const std::array<JointCalibration, kNumJoints> & calibration,
  const std::array<double, kNumJoints> & default_pos,
  const std::array<double, kNumJoints> & last_action,
  const OperatorCommand & command,
  const ImuSample & imu,
  const std::array<double, 2> & phase_features)
{
  // 47-dim OpenDoge_train SuperMimic actor observation:
  //   0:3   base_ang_vel (body)
  //   3:6   projected_gravity (body)
  //   6:18  joint_pos_rel (training convention)
  //   18:30 joint_vel_rel (training convention)
  //   30:42 last_action (raw policy output, no clamp / no sign flip)
  //   42:45 command (vx, vy, yaw_rate)
  //   45:47 gait_phase_sin_cos
  //
  // There is no base_lin_vel: the policy estimates body velocity from the
  // five-frame proprioceptive history inside the ONNX graph.
  ObsFrame obs{};
  std::size_t offset = 0;

  // 0:3 base_ang_vel
  for (std::size_t i = 0; i < 3; ++i) {
    obs[offset + i] = imu.angular_velocity[i];
  }
  offset += 3;

  // 3:6 projected_gravity
  for (std::size_t i = 0; i < 3; ++i) {
    obs[offset + i] = imu.projected_gravity[i];
  }
  offset += 3;

  // 6:18 joint_pos_rel
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    const auto pos = logicalPosition(states[i].position, calibration[i]);
    obs[offset + i] = kJointSign[i] * (pos - default_pos[i]);
  }
  offset += kNumJoints;

  // 18:30 joint_vel_rel (default velocity is zero)
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    obs[offset + i] =
      kJointSign[i] * logicalVelocity(states[i].velocity, calibration[i]);
  }
  offset += kNumJoints;

  // 30:42 last_action
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    obs[offset + i] = last_action[i];
  }
  offset += kNumJoints;

  // 42:45 command
  obs[offset + 0] = command.vx;
  obs[offset + 1] = command.vy;
  obs[offset + 2] = command.yaw_rate;
  offset += 3;

  // 45:47 gait phase
  obs[offset + 0] = phase_features[0];
  obs[offset + 1] = phase_features[1];

  return obs;
}

void HistoryBuffer::reset(const ObsFrame & frame)
{
  for (auto & slot : frames_) {
    slot = frame;
  }
  ready_ = true;
}

void HistoryBuffer::append(const ObsFrame & frame)
{
  for (std::size_t i = 1; i < kObsHistoryLength; ++i) {
    frames_[i - 1] = frames_[i];
  }
  frames_[kObsHistoryLength - 1] = frame;
  ready_ = true;
}

ObsHistory HistoryBuffer::flatten() const
{
  ObsHistory flat{};
  for (std::size_t f = 0; f < kObsHistoryLength; ++f) {
    for (std::size_t i = 0; i < kObsDim; ++i) {
      flat[f * kObsDim + i] = frames_[f][i];
    }
  }
  return flat;
}

}  // namespace opendoge
