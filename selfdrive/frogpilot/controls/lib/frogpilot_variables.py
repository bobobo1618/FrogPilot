from types import SimpleNamespace

from cereal import car
from openpilot.common.conversions import Conversions as CV
from openpilot.common.params import Params

from openpilot.selfdrive.frogpilot.controls.lib.model_manager import RADARLESS_MODELS

class FrogPilotVariables:
  def __init__(self):
    self.frogpilot_toggles = SimpleNamespace()

    self.params = Params()
    self.params_memory = Params("/dev/shm/params")

    self.update_frogpilot_params()

  def update_frogpilot_params(self, started=False):
    toggles = self.frogpilot_toggles

    key = "CarParams" if started else "CarParamsPersistent"
    with car.CarParams.from_bytes(self.params.get(key, block=True)) as msg:
      CP = msg

    toggles.is_metric = self.params.get_bool("IsMetric")
    toggles.radarless_model = self.params.get("Model", block=True, encoding='utf-8') in RADARLESS_MODELS

    toggles.always_on_lateral_pause_speed = self.params.get_int("PauseAOLOnBrake")

    toggles.conditional_experimental_mode = CP.openpilotLongitudinalControl and self.params.get_bool("ConditionalExperimental")
    toggles.conditional_curves = toggles.conditional_experimental_mode and self.params.get_bool("CECurves")
    toggles.conditional_curves_lead = toggles.conditional_curves and self.params.get_bool("CECurvesLead")
    toggles.conditional_limit = self.params.get_int("CESpeed") * (CV.KPH_TO_MS if toggles.is_metric else CV.MPH_TO_MS) if toggles.conditional_experimental_mode else 0
    toggles.conditional_limit_lead = self.params.get_int("CESpeedLead") * (CV.KPH_TO_MS if toggles.is_metric else CV.MPH_TO_MS) if toggles.conditional_experimental_mode else 0
    toggles.conditional_navigation = toggles.conditional_experimental_mode and self.params.get_bool("CENavigation")
    toggles.conditional_navigation_intersections = toggles.conditional_navigation and self.params.get_bool("CENavigationIntersections")
    toggles.conditional_navigation_lead = toggles.conditional_navigation and self.params.get_bool("CENavigationLead")
    toggles.conditional_navigation_turns = toggles.conditional_navigation and self.params.get_bool("CENavigationTurns")
    toggles.conditional_signal = toggles.conditional_experimental_mode and self.params.get_bool("CESignal")
    toggles.conditional_slower_lead = toggles.conditional_experimental_mode and self.params.get_bool("CESlowerLead")
    toggles.conditional_stop_lights = toggles.conditional_experimental_mode and self.params.get_bool("CEStopLights")
    toggles.conditional_stop_lights_lead = toggles.conditional_stop_lights and self.params.get_bool("CEStopLightsLead")

    custom_alerts = self.params.get_bool("CustomAlerts")
    toggles.green_light_alert = custom_alerts and self.params.get_bool("GreenLightAlert")
    toggles.lead_departing_alert = not toggles.radarless_model and custom_alerts and self.params.get_bool("LeadDepartingAlert")
    toggles.loud_blindspot_alert = custom_alerts and self.params.get_bool("LoudBlindspotAlert")

    toggles.custom_personalities = CP.openpilotLongitudinalControl and self.params.get_bool("CustomPersonalities")
    toggles.aggressive_jerk = self.params.get_float("AggressiveJerk")
    toggles.aggressive_follow = self.params.get_float("AggressiveFollow")
    toggles.standard_jerk = self.params.get_float("StandardJerk")
    toggles.standard_follow = self.params.get_float("StandardFollow")
    toggles.relaxed_jerk = self.params.get_float("RelaxedJerk")
    toggles.relaxed_follow = self.params.get_float("RelaxedFollow")
    toggles.traffic_mode_jerk = [self.params.get_float("TrafficJerk"), toggles.aggressive_jerk] if toggles.custom_personalities else [1.0, 0.5]
    toggles.traffic_mode_t_follow = [self.params.get_float("TrafficFollow"), toggles.aggressive_follow] if toggles.custom_personalities else [0.5, 1.0]

    custom_theme = self.params.get_bool("CustomTheme")
    toggles.current_holiday_theme = self.params_memory.get_int("CurrentHolidayTheme") if custom_theme and self.params.get_bool("HolidayThemes") else 0
    toggles.custom_sounds = self.params.get_int("CustomSounds") if custom_theme else 0
    toggles.goat_scream = toggles.custom_sounds == 1 and self.params.get_bool("GoatScream")
    toggles.random_events = custom_theme and self.params.get_bool("RandomEvents")

    custom_ui = self.params.get_bool("CustomUI")
    toggles.adjacent_lanes = custom_ui and self.params.get_bool("AdjacentPath")
    toggles.blind_spot_path = custom_ui and self.params.get_bool("BlindSpotPath")

    device_management = self.params.get_bool("DeviceManagement")
    toggles.increase_thermal_limits = device_management and self.params.get_bool("IncreaseThermalLimits")

    toggles.experimental_mode_via_press = CP.openpilotLongitudinalControl and self.params.get_bool("ExperimentalModeActivation")
    toggles.experimental_mode_via_distance = toggles.experimental_mode_via_press and self.params.get_bool("ExperimentalModeViaDistance")
    toggles.experimental_mode_via_lkas = toggles.experimental_mode_via_press and self.params.get_bool("ExperimentalModeViaLKAS")

    toggles.nudgeless = self.params.get_bool("NudgelessLaneChange")
    toggles.lane_change_delay = self.params.get_int("LaneChangeTime") if toggles.nudgeless else 0
    toggles.lane_detection = toggles.nudgeless and self.params.get_int("LaneDetectionWidth") != 0
    toggles.lane_detection_width = self.params.get_int("LaneDetectionWidth") * (1 if toggles.is_metric else CV.FOOT_TO_METER) / 10 if toggles.lane_detection else 0
    toggles.one_lane_change = toggles.nudgeless and self.params.get_bool("OneLaneChange")

    lateral_tune = self.params.get_bool("LateralTune")
    toggles.force_auto_tune = lateral_tune and self.params.get_float("ForceAutoTune")
    stock_steer_ratio = self.params.get_float("SteerRatioStock")
    toggles.steer_ratio = self.params.get_float("SteerRatio") if lateral_tune else stock_steer_ratio
    toggles.taco_tune = lateral_tune and self.params.get_bool("TacoTune")
    toggles.turn_desires = lateral_tune and self.params.get_bool("TurnDesires")
    toggles.use_custom_steer_ratio = toggles.steer_ratio != stock_steer_ratio

    toggles.long_pitch = CP.openpilotLongitudinalControl and self.params.get_bool("LongPitch")

    longitudinal_tune = CP.openpilotLongitudinalControl and self.params.get_bool("LongitudinalTune")
    toggles.acceleration_profile = self.params.get_int("AccelerationProfile") if longitudinal_tune else 0
    toggles.aggressive_acceleration = longitudinal_tune and self.params.get_bool("AggressiveAcceleration")
    toggles.deceleration_profile = self.params.get_int("DecelerationProfile") if longitudinal_tune else 0
    toggles.increased_stopping_distance = self.params.get_int("StoppingDistance") * (1 if toggles.is_metric else CV.FOOT_TO_METER) if longitudinal_tune else 0
    toggles.lead_detection_threshold = self.params.get_int("LeadDetectionThreshold") / 100 if longitudinal_tune else 0.5
    toggles.smoother_braking = longitudinal_tune and self.params.get_bool("SmoothBraking")
    toggles.smoother_braking_far_lead = toggles.smoother_braking and self.params.get_bool("SmoothBrakingFarLead")
    toggles.smoother_braking_jerk = toggles.smoother_braking and self.params.get_bool("SmoothBrakingJerk")
    toggles.sport_plus = longitudinal_tune and self.params.get_int("AccelerationProfile") == 3
    toggles.traffic_mode = longitudinal_tune and self.params.get_bool("TrafficMode")

    quality_of_life = self.params.get_bool("QOLControls")
    toggles.custom_cruise_increase = self.params.get_int("CustomCruise") if quality_of_life else 1
    toggles.custom_cruise_increase_long = self.params.get_int("CustomCruiseLong") if quality_of_life else 5
    toggles.pause_lateral_below_speed = self.params.get_int("PauseLateralSpeed") * (CV.KPH_TO_MS if toggles.is_metric else CV.MPH_TO_MS) if quality_of_life else 0
    toggles.pause_lateral_below_signal = quality_of_life and self.params.get_bool("PauseLateralOnSignal")
    toggles.reverse_cruise_increase = quality_of_life and self.params.get_bool("ReverseCruise")
    toggles.set_speed_offset = self.params.get_int("SetSpeedOffset") * (1 if toggles.is_metric else CV.MPH_TO_KPH) if quality_of_life else 0

    toggles.map_turn_speed_controller = CP.openpilotLongitudinalControl and self.params.get_bool("MTSCEnabled")
    toggles.mtsc_curvature_check = toggles.map_turn_speed_controller and self.params.get_bool("MTSCCurvatureCheck")
    self.params_memory.put_float("MapTargetLatA", 2 * (self.params.get_int("MTSCAggressiveness") / 100))

    toggles.sng_hack = CP.openpilotLongitudinalControl and self.params.get_bool("SNGHack")

    toggles.speed_limit_controller = CP.openpilotLongitudinalControl and self.params.get_bool("SpeedLimitController")
    toggles.force_mph_dashboard = toggles.speed_limit_controller and self.params.get_bool("ForceMPHDashboard")
    toggles.map_speed_lookahead_higher = self.params.get_int("SLCLookaheadHigher") if toggles.speed_limit_controller else 0
    toggles.map_speed_lookahead_lower = self.params.get_int("SLCLookaheadLower") if toggles.speed_limit_controller else 0
    toggles.offset1 = self.params.get_int("Offset1") if toggles.speed_limit_controller else 0
    toggles.offset2 = self.params.get_int("Offset2") if toggles.speed_limit_controller else 0
    toggles.offset3 = self.params.get_int("Offset3") if toggles.speed_limit_controller else 0
    toggles.offset4 = self.params.get_int("Offset4") if toggles.speed_limit_controller else 0
    toggles.set_speed_limit = toggles.speed_limit_controller and self.params.get_bool("SetSpeedLimit")
    toggles.speed_limit_alert = toggles.speed_limit_controller and self.params.get_bool("SpeedLimitChangedAlert")
    toggles.speed_limit_confirmation = toggles.speed_limit_controller and self.params.get_bool("SLCConfirmation")
    toggles.speed_limit_confirmation_lower = toggles.speed_limit_confirmation and self.params.get_bool("SLCConfirmationLower")
    toggles.speed_limit_confirmation_higher = toggles.speed_limit_confirmation and self.params.get_bool("SLCConfirmationHigher")
    toggles.speed_limit_controller_override = self.params.get_int("SLCOverride") if toggles.speed_limit_controller else 0
    toggles.use_experimental_mode = self.params.get_int("SLCFallback") == 2 if toggles.speed_limit_controller else False
    toggles.use_previous_limit = self.params.get_int("SLCFallback") == 1 if toggles.speed_limit_controller else False
    toggles.speed_limit_priority1 = self.params.get("SLCPriority1", encoding='utf-8') if toggles.speed_limit_controller else None
    toggles.speed_limit_priority2 = self.params.get("SLCPriority2", encoding='utf-8') if toggles.speed_limit_controller else None
    toggles.speed_limit_priority3 = self.params.get("SLCPriority3", encoding='utf-8') if toggles.speed_limit_controller else None
    toggles.speed_limit_priority_highest = toggles.speed_limit_priority1 == "Highest"
    toggles.speed_limit_priority_lowest = toggles.speed_limit_priority1 == "Lowest"

    toyota_doors = self.params.get_bool("ToyotaDoors")
    toggles.lock_doors = toyota_doors and self.params.get_bool("LockDoors")
    toggles.unlock_doors = toyota_doors and self.params.get_bool("UnlockDoors")

    toggles.use_ev_tables = CP.openpilotLongitudinalControl and self.params.get_bool("EVTable")

    toggles.vision_turn_controller = CP.openpilotLongitudinalControl and self.params.get_bool("VisionTurnControl")
    toggles.curve_sensitivity = self.params.get_int("CurveSensitivity") / 100 if toggles.vision_turn_controller else 1
    toggles.turn_aggressiveness = self.params.get_int("TurnAggressiveness") / 100 if toggles.vision_turn_controller else 1

FrogPilotVariables = FrogPilotVariables()
FrogPilotToggles = FrogPilotVariables.frogpilot_toggles
