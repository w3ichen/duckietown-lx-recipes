from duckietown_world import (
    get_DB18_nominal,
    PlatformDynamics,
    PWMCommands,
    SampledSequenceBuilder,
    SE2Transform,
    DynamicModel,
)

import numpy as np
import geometry as geo

NOMINAL_WHEEL_RADIUS = 0.0318
NOMINAL_BASELINE = 0.1
NOMINAL_ENCODER_TICKS = 135


def get_wheel_speed(omega, v_a, baseline=NOMINAL_BASELINE, radius=NOMINAL_WHEEL_RADIUS):

    # Using the inverse kinematics we obtain the angular velocities of the two wheels
    omega_l = (v_a - 0.5 * omega * baseline) / radius
    omega_r = (v_a + 0.5 * omega * baseline) / radius
    return omega_l, omega_r


def pwm_commands_from_PID(omega, v_a, k_r=27, k_l=27, limit=1.0):
    """
    Function returning a PWMCommands object given the control inputs
    u = [omega, v_a]
    Input:
        - omega:    commanded angular velocity in rad/s
        - v_a:      commanded linear velcoity in m/s
        - k_r:      right motor constant
        - k_r:      right motor constant
        - limit:    maximum wheel speed (in percentage 0-1)
    Output:
        - PWMCommands object
    """

    omega_l, omega_r = get_wheel_speed(omega, v_a)

    # This gives us the duty cycle input to each motor (CLIPPED TO [-1,+1])
    u_l = omega_l / k_l
    u_r = omega_r / k_r

    u_l = np.clip(u_l, a_min=-limit, a_max=limit)
    u_r = np.clip(u_r, a_min=-limit, a_max=limit)

    return PWMCommands(motor_left=u_l, motor_right=u_r)


def get_measured_ticks(model: DynamicModel) -> tuple([int, int]):
    ticks_left = model.axis_left_obs_rad / model.parameters.encoder_resolution_rad
    ticks_right = model.axis_right_obs_rad / model.parameters.encoder_resolution_rad

    return ticks_left, ticks_right


def integrate_dynamics(
    initial_pose, initial_vel, y_ref, controller, odometry_function=None, delta_phi=None
):
    """
    Input:
        - inital_pose: 3 elements list containing the initial position and orientation
                        [x_0,y_0,theta_0], theta_0 in degrees
        - initial_vel: 2 elements list with initial linear and angular vel [v_0, omega_0]
        - y_ref: the target y position
    """

    # initial pose and velocity
    initial_pose[2] = np.deg2rad(initial_pose[2])
    v = initial_vel[0]
    omega = initial_vel[1]

    q0 = geo.SE2_from_xytheta(initial_pose)
    v0 = geo.se2_from_linear_angular(
        np.array([v * np.cos(initial_pose[2]), v * np.sin(initial_pose[2])]), omega
    )

    c0 = q0, v0

    initial_time = 0.0
    timestep = 0.1
    t_max = 60
    n = int(t_max / timestep)

    nominal_duckie = get_DB18_nominal(delay=0)
    state = nominal_duckie.initialize(c0=c0, t0=initial_time)

    ssb: SampledSequenceBuilder[PlatformDynamics] = SampledSequenceBuilder[
        PlatformDynamics
    ]()
    ssb.add(initial_time, state)

    # Set integrator state
    e_int = 0
    e = initial_pose[1] - y_ref

    # Set the commanded parameters
    v_0 = 0.22
    prev_e_y = 0.0
    prev_int_y = 0.0

    # Initialize odometry variables
    if odometry_function is not None:
        x_hat, y_hat, theta_hat = initial_pose[0:3]
        prev_ticks_left = prev_ticks_right = 0
        ticks_left = ticks_right = 0
        
    for i in range(n):
        # Get y_hat from pose
        last_pose, last_vel = state.TSE2_from_state()

        if odometry_function is None:
            y_hat = last_pose[1][2]

        else:
            assert (delta_phi is not None, "Need to pass a delta_phi function!")
            # Get measured ticks from dynamics
            prev_ticks_left = ticks_left
            prev_ticks_right = ticks_right

            ticks_left, ticks_right = get_measured_ticks(state)

            delta_phi_left, ticks_left = delta_phi(
                ticks_left, prev_ticks_left, NOMINAL_ENCODER_TICKS
            )
            delta_phi_right, ticks_right = delta_phi(
                ticks_right, prev_ticks_right, NOMINAL_ENCODER_TICKS
            )

            x_hat, y_hat, theta_hat = odometry_function(
                R=NOMINAL_WHEEL_RADIUS,
                baseline=NOMINAL_BASELINE,
                x_prev=x_hat,
                y_prev=y_hat,
                theta_prev=theta_hat,
                delta_phi_left=delta_phi_left,
                delta_phi_right=delta_phi_right,
            )

        v_0, omega, e, e_int = controller(
            v_0, y_ref, y_hat, prev_e_y, prev_int_y, delta_t=timestep
        )
        prev_e_y = e
        prev_int_y = e_int

        # Simulate driving
        commands = pwm_commands_from_PID(omega, v_0)

        state = state.integrate(timestep, commands)

        t = initial_time + (i + 1) * timestep

        ssb.add(t, state)

    return ssb.as_sequence()
