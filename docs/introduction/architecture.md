---
layout: "contents"
title: Architecture
firstpage:
---

# Architecture

```{mermaid}

flowchart LR
    A[Python Gym Environment] --> |TCP| B[Bevy Server]
    B --> |Controls| C[Game Engine]
    C --> |Physics| D[Aircraft State]
    D --> |Updates| B
    B --> |Observations| A
```

### Core Structure

The system has three main parts:
1. Python Gym Environment (AI interface)
2. Bevy Server (communication handler)
3. Game Engine (physics simulation)

```{mermaid}

    sequenceDiagram
        participant Agent
        participant Gym
        participant Server
        participant Engine

        Agent->>Gym: take_action()
        Gym->>Server: Send action via TCP
        Server->>Engine: Update controls
        Engine->>Engine: Run physics
        Engine->>Server: Update state
        Server->>Gym: Send observation
        Gym->>Agent: return observation
```

### Key Operations

```{mermaid}
stateDiagram-v2
    [*] --> Starting
    Starting --> Initializing: Launch Server
    Initializing --> Ready: TCP Connected
    Ready --> Running: First Action
    Running --> Running: Step
    Running --> Resetting: Reset Command
    Resetting --> Ready
    Running --> Closing: Close Command
    Ready --> Closing: Close Command
    Closing --> [*]

```

#### Starting Up
1. Python launches the Bevy server executable
2. Server starts and opens a TCP port
3. Python connects to the server
4. Server initializes the game engine
5. System is ready for commands

#### Taking Actions
1. Agent sends action through Gym interface
2. Python converts action to control commands
3. Commands sent to server via TCP
4. Server updates aircraft controls
5. Physics engine simulates movement
6. New state sent back as observations

#### State Management
1. Aircraft state stored in shared memory
2. Action queue holds pending commands
3. State buffer stores current positions
4. TCP messages synchronize everything

## Technical Details

### Command Structure
```json
{
    "command_type": {
        "data": {...}
    }
}
```

### Main Commands:
- Initialize: Set up simulation
- Step: Process actions
- Reset: Start new episode
- Close: Clean up

### State Tracking
- Position (x, y, z)
- Rotation (roll, pitch, yaw)
- Velocities
- Control surfaces
- Engine state

### Key Files
- `abstract.py`: Python Gym interface
- `bevy_server.rs`: Communication handler
- Action/observation handlers

## Common Operations

### Creating Environment
```python
env = FlyerEnv(config={
    "aircraft_type": "dubins",
    "max_steps": 1000
})
```

### Taking Steps
```python
action = agent.get_action()
obs, reward, done, info = env.step(action)
```

### Cleanup
```python
env.close()  # Shuts down server and engine
```

## Error Handling

The system handles:
1. Connection loss
2. Invalid actions
3. Physics errors
4. Resource cleanup

## Performance Notes

For best performance:
- Keep step rate consistent
- Clean up resources
- Monitor memory usage
- Check TCP connection health