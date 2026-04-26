clc
close all
clear


%% 
load("project2verification.mat");

%% 
const = getConst();

% Initial conditions
% Initial positions and velocities
x0 = 0;                 % Horizontal position (m)
z0 = 0.25;              % Vertical position (m)
vx0 = 0;                % Horizontal velocity (m/s)
vz0 = 0;                % Vertical velocity (m/s)

% Initial mass calculations
initial_air_mass = const.p0 * (const.VB - const.Vol_w_i) / (const.Rair * const.T0);
initial_rocket_mass = const.mB + (const.rho_w * const.Vol_w_i) + initial_air_mass;

% State vector: [x, vx, z, vz, mr, V_air, mair]
initialState = [x0; vx0; z0; vz0; initial_rocket_mass; (const.VB -const.Vol_w_i); initial_air_mass];

% Time span for simulation
tspan = [0, 5]; % Simulate for 5 seconds

% Solve the dynamics using ode45
[t, state] = ode45(@(t, state) rocketDynamics_ode(t, state, const), tspan, initialState);
[dState, thrust] = rocketDynamics(t, state, const);

for i = 1:length(t)
    [dState, thrust(i)] = rocketDynamics(t(i), state(i,:), const);
end

% Extract results
x = state(:, 1); % Horizontal position
vx = state(:,2);
z = state(:, 3); % Vertical position
vz = state(:,4);
v_air = state(:,6);

% Plot trajectory
figure;
plot(x, z, 'r-');
hold on
plot(verification.distance,verification.height, 'b-');
xlabel('Horizontal Distance (m)');
ylabel('Vertical Distance (m)');
title('Rocket Trajectory');
legend('modeled','actual')
grid on;

figure;
plot(t, v_air, 'r-');
hold on
plot(verification.time,verification.volume_air, 'b-');
xlabel('Time (s)');
ylabel('Pressure (Pa)');
title(' Volume of Air over time');
legend('modeled','actual')
grid on;

figure;
plot(t, vz, 'r-');
hold on;
plot(verification.time,verification.velocity_y, 'b-');
xlabel('Time (s)');
ylabel('Vertical Velocity (m/s)');
title('Vertical Velocity Over Time');
legend('modeled','actual')
grid on;

% Plot vertical velocity vs time
figure;
plot(t, vx, 'r-');
hold on;
plot(verification.time,verification.velocity_x, 'b-');
xlabel('Time (s)');
ylabel('Horizontal Velocity (m/s)');
title('Horizontal Velocity Over Time');
legend('modeled','actual')
grid on;


figure;
plot(t, thrust, 'r-');
hold on
plot(verification.time,verification.thrust, 'b-');
xlabel('Time (s)');
ylabel('Thrust (N)');
title('Thrust Over Time');
legend('modeled','actual')
grid on;
xlim([0 0.2]);
ylim([0 200]);


maxThrust = max(thrust);
maxHeight = max(z);
maxDistance = max(x);


%% Function to give to ODE45
function dState = rocketDynamics_ode(t, state, constants)
    [dState, ~] = rocketDynamics(t, state, constants);
end


%% rocket function
function [dState, thrust] = rocketDynamics(t, state, constants)
    % Unpack state variables
    x = state(1);      % Horizontal position
    vx = state(2);     % Horizontal velocity
    z = state(3);      % Vertical position
    vz = state(4);     % Vertical velocity
    mr = state(5);     % Rocket mass
    V_air = state(6);  % Air volume
    m_air = state(7);   % Air mass\
    m_air0 = constants.p0*(constants.VB - constants.Vol_w_i)/(constants.Rair*constants.T0);
    
    % Initialize derivatives
    dx = vx;
    dz = vz;
    
    % Calculate velocity magnitude and heading vector
    v = sqrt(vx^2 + vz^2); % Magnitude of velocity
    if sqrt(x^2+(z-constants.z0)^2) <= constants.ls
        hx = cosd(constants.theta_i);
        hz = sind(constants.theta_i);
    else
        hx = vx / v; % Horizontal component of heading
        hz = vz / v; % Vertical component of heading
    end

    % Air pressure calculation
    p1 = constants.p0 * ((constants.VB - constants.Vol_w_i)/V_air)^constants.gamma; % Air pressure
    p_end = (constants.p0)*((constants.VB-constants.Vol_w_i)/(constants.VB))^constants.gamma;
    p2 = p_end*(m_air/m_air0)^constants.gamma;

    % Check current phase
    if V_air < constants.VB
        % Phase 1: Water Expulsion
        ve = sqrt(2 * (p1 - constants.pa) / constants.rho_w); % Exit velocity of water
        At = pi * (constants.de / 2)^2; % Throat area
        mdot_w = constants.cdis * constants.rho_w * At * ve; % Water mass flow rate
        thrust = mdot_w * ve; % Thrust magnitude
        mdot_r = -mdot_w; % Rocket mass loss rate
        dV_air = constants.cdis * At * ve; % Air volume rate of change
        mdot_air = 0;
        
    elseif p2 > constants.pa
        % Phase 2: Air Expulsion
        rho = m_air/constants.VB;
        T = p2/(rho*constants.Rair);
        p_critical = p2 * (2 / (constants.gamma + 1))^(constants.gamma / (constants.gamma - 1));
        if p_critical > constants.pa
            % Choked flow
            Te = T * (2 / (constants.gamma + 1)); % Exit temperature
            pe = p_critical;
            rho_e = pe/(constants.Rair*Te);
            ve = sqrt(constants.gamma * constants.Rair * Te); % Exit velocity
        elseif p_critical <= constants.pa
            % Unchoked flow: Use algebraic solution for Mach number
            Me = sqrt((2 / (constants.gamma-1)) * ((p2/constants.pa)^((constants.gamma-1)/constants.gamma)-1)); % Mach number
            pe = constants.pa;
            Te = T / (1 + ((constants.gamma - 1) / 2) * Me^2); % Exit temperature
            rho_e = pe/(constants.Rair*Te);
            ve = Me * sqrt(constants.gamma * constants.Rair * Te); % Exit velocity
        end
        
      
        At = pi * (constants.de / 2)^2; % Throat area
        mdot_air = -constants.cdis * rho_e * At * ve; % Air mass flow rate
        thrust = -mdot_air * ve + (pe - constants.pa) * At; % Thrust magnitude
        mdot_r = mdot_air; % Rocket mass loss rate
        dV_air = 0; % Air volume stays constant in Phase 2
        
    else
        % Phase 3: Ballistic Flight
        thrust = 0; % No thrust
        mdot_r = 0; % No mass change
        dV_air = 0; % No volume change
        mdot_air = 0;
    end
    
    % Compute thrust components using heading vector
    thrust_x = thrust * hx;
    thrust_z = thrust * hz;

    % Drag force components
    drag = 0.5 * constants.rho_air * v^2 * constants.CD * (pi * (constants.dB / 2)^2); % Drag magnitude
    drag_x = drag * hx;
    drag_z = drag * hz;
    
    % Accelerations
    ax = (thrust_x - drag_x) / mr;
    az = (thrust_z - drag_z - mr * constants.g) / mr;
    
    % Derivatives
    dState = [dx; ax; dz; az; mdot_r; dV_air; mdot_air];
    
    if z<= 0 
        thrust = 0;
        az = 0;
        ax = 0;
        drag = 0;
        vx = 0;
        vz = 0;
         dState = [0; 0; 0; 0; 0; 0; 0];
    end

end



%% Constant Function
function const = getConst()
    % Physical constants
    const.g = 9.81;               % Acceleration due to gravity (m/s^2)
    const.cdis = 0.78;            % Discharge coefficient
    const.rho_air = 0.961;        % Ambient air density (kg/m^3)
    const.rho_w = 1000;           % Water density (kg/m^3)
    const.Rair = 287;             % Specific gas constant for air (J/(kg·K))
    const.gamma = 1.4;            % Ratio of specific heats for air

    % Rocket dimensions
    const.VB = 0.002;             % Volume of empty bottle (m^3)
    const.de = 2.1 / 100;         % Diameter of the throat (exit) (m)
    const.dB = 10.5 / 100;        % Diameter of the bottle (m)

    % Initial conditions
    const.mB = 0.15;              % Mass of empty bottle with cone and fins (kg)
    const.Vol_w_i = 0.0005;        % Initial water volume inside the bottle (m^3)
    const.T0 = 310;               % Initial temperature of air (K)
    const.theta_i = 40;           % Launch angle (degrees)

    % Pressure values
    const.pa = 12.1 * 6894.76;    % Atmospheric pressure (converted from psia to Pa)
    const.p0 = 330948 + const.pa; % Initial absolute pressure (converted from psig to Pa)

    % Aerodynamic properties
    const.CD = 0.425;             % Drag coefficient

    % Miscellaneous
    const.ls = 0.5;               % Length of the test stand (m)
    const.z0 = 0.25;
end

