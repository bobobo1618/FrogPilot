using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# you can rename the struct, but don't change the identifier
struct FrogPilotCarControl @0x81c2f05a394cf4af {
  alwaysOnLateral @0 :Bool;
  speedLimitChanged @1 :Bool;
  trafficModeActive @2 :Bool;
}

struct FrogPilotCarState @0xaedffd8f31e7b55d {
  struct ButtonEvent {
    enum Type {
      lkas @0;
    }
  }

  dashboardSpeedLimit @0 :Float32;
  ecoGear @1 :Bool;
  sportGear @2 :Bool;
}

struct FrogPilotDeviceState @0xf35cc4560bbf6ec2 {
  freeSpace @0 :Int16;
  usedSpace @1 :Int16;
}

struct FrogPilotNavigation @0xda96579883444c35 {
  approachingIntersection @0 :Bool;
  approachingTurn @1 :Bool;
}

struct FrogPilotPlan @0x80ae746ee2596b11 {
  accelerationJerk @0 :Float32;
  accelerationJerkStock @1 :Float32;
  adjustedCruise @2 :Float64;
  conditionalExperimental @3 :Bool;
  desiredFollowDistance @4 :Int16;
  egoJerk @5 :Float32;
  egoJerkStock @6 :Float32;
  jerk @7 :Float32;
  laneWidthLeft @8 :Float32;
  laneWidthRight @9 :Float32;
  minAcceleration @10 :Float32;
  maxAcceleration @11 :Float32;
  redLight @12 :Bool;
  safeObstacleDistance @13 :Int16;
  safeObstacleDistanceStock @14 :Int16;
  slcOverridden @15 :Bool;
  slcOverriddenSpeed @16 :Float64;
  slcSpeedLimit @17 :Float64;
  slcSpeedLimitOffset @18 :Float32;
  stoppedEquivalenceFactor @19 :Int16;
  tFollow @20 :Float32;
  unconfirmedSlcSpeedLimit @21 :Float64;
  vCruise @22 :Float32;
  vtscControllingCurve @23 :Bool;
}

struct CustomReserved5 @0xa5cd762cd951a455 {
}

struct CustomReserved6 @0xf98d843bfd7004a3 {
}

struct CustomReserved7 @0xb86e6369214c01c8 {
}

struct CustomReserved8 @0xf416ec09499d9d19 {
}

struct CustomReserved9 @0xa1680744031fdb2d {
}
