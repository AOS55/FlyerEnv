import pyflyer
import signal
import os


def from_dict():
    
    config = {
        # Simulation parameters
        "seed": 42,
        "max_episode_steps": 1000,
        "steps_per_action": 4,
        "time_step": 1.0/60.0,  # 60 Hz simulation
        
        # Aircraft configurations
        "aircraft_config": [
            # First aircraft - Dubins aircraft with continuous actions
            {
                "type": "dubins",
                "name": "scout_aircraft",
                "action_type": "Continuous",
                "observation_type": "Continuous",
                "config": {
                    "max_speed": 30.0,
                    "min_speed": 15.0,
                    "acceleration": 2.0,
                    "max_bank_angle": 45.0,  # degrees
                    "max_turn_rate": 3.0,
                    "max_climb_rate": 5.0,
                    "max_descent_rate": 3.0,
                    "random_start": {
                        "origin_x": 0.0,
                        "origin_y": 0.0,
                        "variance": 100.0,
                        "min_altitude": 100.0,
                        "max_altitude": 200.0
                    }
                }
            },
            
            # Second aircraft - Full aircraft model with discrete actions
            {
                "type": "full",
                "name": "twin_otter",
                "action_type": "Discrete",
                "observation_type": "Continuous",
                "config": {
                    "ac_type": "twin_otter",
                    "mass": {
                        "mass": 4500.0,  # kg
                        "ixx": 25000.0,  # kg*m^2
                        "iyy": 35000.0,
                        "izz": 28000.0,
                        "ixz": 1000.0
                    },
                    "geometry": {
                        "wing_area": 39.0,  # m^2
                        "wing_span": 19.8,  # m
                        "mac": 2.0  # mean aerodynamic chord (m)
                    }
                }
            }
        ],
        
        # Terrain configuration
        "terrain_config": {
            "noise": {
                "height": {
                    "scale": 800.0,
                    "octaves": 6,
                    "persistence": 0.5,
                    "lacunarity": 2.0,
                    "layers": [
                        {
                            "scale": 100.0,
                            "amplitude": 1.0,
                            "octaves": 1,
                            "persistence": 0.5,
                            "weight": 1.0
                        },
                        {
                            "scale": 50.0,
                            "amplitude": 0.5,
                            "octaves": 1,
                            "persistence": 0.5,
                            "weight": 0.5
                        }
                    ]
                }
            },
            "biome": {
                "thresholds": {
                    "water": 0.45,
                    "mountain_start": 0.75,
                    "mountain_width": 0.1,
                    "beach_width": 0.025,
                    "forest_moisture": 0.95,
                    "desert_moisture": 0.2,
                    "field_sizes": [96.0, 128.0, 256.0, 512.0]
                }
            }
        }
    }
    
    pyflyer.FlyerEnv(config)


def sigint_handler(signum, frame):
    print("\nCaught SIGINT, exiting gracefully...")
    os._exit(1)


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    
    config = {"aircraft_config": [
                # First aircraft - Dubins aircraft with continuous actions
                {
                    "type": "dubins",
                    "name": "scout_aircraft",
                    "action_type": "Continuous",
                    "observation_type": "Continuous",
                    "config": {
                        "max_speed": 30.0,
                        "min_speed": 15.0,
                        "acceleration": 2.0,
                        "max_bank_angle": 45.0,  # degrees
                        "max_turn_rate": 3.0,
                        "max_climb_rate": 5.0,
                        "max_descent_rate": 3.0,
                        "random_start": {
                            "origin_x": 0.0,
                            "origin_y": 0.0,
                            "variance": 100.0,
                            "min_altitude": 100.0,
                            "max_altitude": 200.0
                        }
                    }
                }
            ]
        }
    
    try:
        env = pyflyer.FlyerEnv(config, render_mode="human")
        print(f"env: {env}")
        action = {}
        for _ in range(100):
            print("about to step")
            env.step(action)
            print("stepped")
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        os._exit(1)
     
    
if __name__ == "__main__":
    main()