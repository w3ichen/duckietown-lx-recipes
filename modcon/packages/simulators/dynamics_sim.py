from duckietown_world import (
    get_DB18_nominal,
    PlatformDynamics,
    PWMCommands,
    SampledSequenceBuilder,
)

import numpy as np
import geometry as geo

def get_wheel_speed(omega,v_a):
    # These are the default parameters
    baseline = 0.1  # baseline = 2L
    R = 0.0318

    # Using the inverse kinematics we obtain the angular velocities of the two wheels
    omega_l = (v_a - 0.5*omega*baseline)/R
    omega_r = (v_a + 0.5*omega*baseline)/R
    return omega_l, omega_r

def pwm_commands_from_PID(omega,v_a):
    omega_l, omega_r = get_wheel_speed(omega,v_a)

    # Motor constant    
    k = 27
    limit = 1.0

    # This gives us the duty cycle input to each motor (CLIPPED TO [-1,+1])
    u_l = omega_l/k
    u_r = omega_r/k

    u_l = np.clip(u_l, a_min=-limit, a_max=limit)
    u_r = np.clip(u_r, a_min=-limit, a_max=limit)

    return PWMCommands(motor_left=u_l,motor_right=u_r)


    
def integrate_dynamics(initial_pose, initial_vel, y_ref):
    """
    Input:
        - inital_pose: 3 elements list containing the initial position and orientation 
                        [x_0,y_0,theta_0], theta_0 in degrees
        - initial_vel: 2 elements list with initial linear and angular vel [v_0, omega_0]
        - y_ref: the target y position
    """

    # initial pose and velocity
    initial_pose[2]=np.deg2rad(initial_pose[2])
    v = initial_vel[0]
    omega = initial_vel[1]

    q0 = geo.SE2_from_xytheta(initial_pose)
    v0 = geo.se2_from_linear_angular(
        np.array([v*np.cos(initial_pose[2]),v*np.sin(initial_pose[2])]),
        omega
        )

    c0 = q0, v0

    initial_time=0.0
    timestep = 0.1
    t_max = 60
    n = int(t_max / timestep)

    nominal_duckie = get_DB18_nominal(delay=0)
    state = nominal_duckie.initialize(c0=c0,t0=initial_time)

    ssb: SampledSequenceBuilder[PlatformDynamics] = SampledSequenceBuilder[PlatformDynamics]()
    ssb.add(initial_time, state)

    # Set integrator state
    e_int = 0
    e = initial_pose[1]-y_ref

    # Set the commanded parameters
    v_0 = 0.22
    prev_e_y = 0.0
    prev_int_y = 0.0

    for i in range(n):

        # Get y_hat from pose
        last_pose, last_vel = state.TSE2_from_state()
        y_hat = last_pose[1][2]
        
        v_0, omega, e, e_int = PIDController(v_0, y_ref, y_hat, prev_e_y, prev_int_y, delta_t=timestep)
        prev_e_y = e
        prev_int_y = e_int

        # Simulate driving
        commands = pwm_commands_from_PID(omega,v_0)

        state=state.integrate(timestep,commands)

        t = initial_time + (i + 1) * timestep

        ssb.add(t, state)
    
    return ssb.as_sequence()
