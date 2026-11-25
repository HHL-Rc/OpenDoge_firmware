#pragma once

#include <array>

#include "opendoge_deploy/types.hpp"

namespace opendoge
{

/// Gait phase features for the OpenDoge_train SuperMimic actor:
///   steps = round(elapsed_s / control_dt)
///   p     = (steps mod round(period_s/control_dt)) / round(period_s/control_dt)
///   moving = hypot(vx, vy) + abs(yaw_rate) > threshold
///   features = moving ? [sin(2*pi*p), cos(2*pi*p)] : [0, 0]
/// The clock is never reset by the standing gate; it is reset when the
/// deployment enters ActiveRL.
std::array<double, 2> gaitPhaseFeatures(
  double elapsed_s,
  double control_dt,
  double period_s,
  double threshold,
  const OperatorCommand & command);

/// Build one 47-dim OpenDoge_train SuperMimic actor frame:
///   base_ang_vel(3) + projected_gravity(3)
///   + joint_pos_rel(12) + joint_vel_rel(12) + last_action(12)
///   + command(3) + gait_phase_sin_cos(2)
/// Joint terms are expressed in the training sign convention via kJointSign.
/// `default_pos` is in firmware URDF convention.
ObsFrame buildObservationFrame(
  const std::array<MotorState, kNumJoints> & states,
  const std::array<JointCalibration, kNumJoints> & calibration,
  const std::array<double, kNumJoints> & default_pos,
  const std::array<double, kNumJoints> & last_action,
  const OperatorCommand & command,
  const ImuSample & imu,
  const std::array<double, 2> & phase_features);

/// Five-frame, oldest-first history buffer. reset() repeats the initial frame
/// (ONNX metadata history_reset=repeat_initial_frame); append() shifts left.
class HistoryBuffer
{
public:
  void reset(const ObsFrame & frame);
  void append(const ObsFrame & frame);
  ObsHistory flatten() const;
  bool ready() const { return ready_; }
  void clear() { ready_ = false; }

private:
  std::array<ObsFrame, kObsHistoryLength> frames_{};
  bool ready_{false};
};

}  // namespace opendoge
