import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

def plot_aircraft_state(csv_path):
    # Read data
    df = pd.read_csv(csv_path)

    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Aircraft State Analysis', fontsize=16)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Plot 1: Velocities
    ax = axes[0, 0]
    ax.plot(df['time'], df['vx'], 'b-', label='Vx')
    ax.plot(df['time'], df['vy'], 'g-', label='Vy')
    ax.plot(df['time'], df['vz'], 'r-', label='Vz')
    ax.set_title('Velocity Components')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.grid(True)
    ax.legend()

    # Plot 2: Flight Path
    ax = axes[0, 1]
    ax.plot(df['time'], df['airspeed'], 'b-', label='Airspeed')
    ax.plot(df['time'], df['altitude'], 'r-', label='Altitude')
    ax.set_title('Flight Parameters')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()

    # Plot 3: Angles
    ax = axes[1, 0]
    ax.plot(df['time'], df['alpha'], 'b-', label='Alpha')
    ax.plot(df['time'], df['beta'], 'r-', label='Beta')
    ax.set_title('Aerodynamic Angles')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.grid(True)
    ax.legend()

    # Plot 4: Attitude
    ax = axes[1, 1]
    ax.plot(df['time'], df['phi'], 'r-', label='Roll')
    ax.plot(df['time'], df['theta'], 'g-', label='Pitch')
    ax.plot(df['time'], df['psi'], 'b-', label='Yaw')
    ax.set_title('Attitude Angles')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.grid(True)
    ax.legend()

    # Plot 5: Forces
    ax = axes[2, 0]
    ax.plot(df['time'], df['force_x'], 'b-', label='Fx')
    ax.plot(df['time'], df['force_y'], 'g-', label='Fy')
    ax.plot(df['time'], df['force_z'], 'r-', label='Fz')
    ax.set_title('Forces')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.grid(True)
    ax.legend()

    # Plot 6: Moments
    ax = axes[2, 1]
    ax.plot(df['time'], df['moment_x'], 'b-', label='Mx')
    ax.plot(df['time'], df['moment_y'], 'g-', label='My')
    ax.plot(df['time'], df['moment_z'], 'r-', label='Mz')
    ax.set_title('Moments')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Moment (N·m)')
    ax.grid(True)
    ax.legend()

    # Save plot
    plt.savefig('aircraft_state.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as aircraft_state.png")

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Get CSV path from temp directory
    csv_path = os.path.join(tempfile.gettempdir(), "aircraft_state_log.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV file at {csv_path}")
        exit(1)

    plot_aircraft_state(csv_path)
