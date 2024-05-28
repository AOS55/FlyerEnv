(aircraft-controller)=

# Control

All control occurs within the {py:class}`~flyer_env.aircraft.controller.ControlledAircraft` and is broken 
down into low-level loop-closure based PID controllers and high-level control. Each allowing for increasing level of 
abstraction in the action space.  -->

## Low-level control

Low-level controllers are used to hold and maintain the Kinematic aircraft in a fixed attitude and speed. Each 
follow a PID type controller.

```{eval-rst}
.. mermaid:: pid.mmd
```

Controllers include:

- Pitch {py:class}`~flyer_env.aircraft.controller.ControlledAircraft.pitch_controller`
- Roll {py:class}`~flyer_env.aircraft.controller.ControlledAircraft.roll_controller`
- Speed {py:class}`~flyer_env.aircraft.controller.ControlledAircraft.speed_controller`

# High-level control

High-level controllers are used to control higher order functions allowing offloading the problem of decision-making 
to be a higher-order task. These build on the low level controllers to include:

- Altitude {py:class}`~flyer_env.aircraft.controller.ControlledAircraft.alt_controller`
- Heading {py:class}`~flyer_env.aircraft.controller.ControlledAircraft.heading_controller`
- Pursuit {py:class}`~flyer_env.aircraft.controller.ControlledKinematicVehicle.pursuit_controller`

## API

```{eval-rst}
.. automodule:: flyer_env.aircraft.controller
    :members:
```
