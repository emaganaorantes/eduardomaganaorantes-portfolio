%% reading the orbit files
[t_dy,r1, v1] = ReadGFO_Orbit('GNV1B_2024-02-22_C_04-1.txt');

[t_dy,r2, v2] = ReadGFO_Orbit('GNV1B_2024-02-22_D_04.txt');

%% rho function
Rho = GFO_Range(r1,r2);

%% rho dot function 
Rho_dot = GFO_RangeRate(r1,r2,v1,v2);

%% Num diff function 
dt = 1;
[Rho_dot_ND, Rho_dot_diff] = GFO_NUmDiff(dt, Rho, Rho_dot);

%% satellite visibility from maryland slr station
rG = [1130714.219 -4831369.903 3994085.962];
[el, index_vis] = SatVisibility(rG, r1);

%% output to csv file 
WriteGFO_CSV('GFO_file.csv', t_dy, Rho, Rho_dot)

%% plots
N = 86400;

%here we convert to our correct units
x = 1/36000:(1/3600):24;
x = x';

rho_km = Rho / 1000;

figure;

% rho plot
subplot(3,1,1);
plot(x, rho_km', 'b');
xlabel('time [hours in 22 Feb. 2024]');
ylabel('range [km]');
grid on;

%%


% rho dot
subplot(3,1,2);
plot(x, Rho_dot, 'b');
xlabel('time [hours in 22 Feb. 2024]');
ylabel('range-rate [m/s]');
grid on;

% plot for el 
subplot(3,1,3);
plot(x, el, 'b');
hold on;
plot(x,index_vis,'r'); %% this is where we put the plot of elevation angles greater than 10
hold on;
xlabel('time [hours in 22 Feb. 2024]');
ylabel('elevation angle [deg]');
grid on;

%% read GRACE-FO orbit data
filename1 = 'GNV1B_2024-02-22_C_04-1.txt';
[t_dy, r1, v1] = ReadGFO_Orbit(filename1);
x = r1(1:1:end,1);
y = r1(1:1:end,2);
z = r1(1:1:end,3);

%% Convert the cratesian to sphberical coordinates 
xyz = [x y z];
lla = ecef2lla(r1);
llaG = ecef2lla(rG);
lat = lla(:,1);
lon = lla(:,2);
h = lla(:,3);

latOver10 = lat;
lonOver10 = lon;
latOver10(isnan(index_vis)) = nan;
lonOver10(isnan(index_vis)) = nan;
%%
figure;
geoplot(lat, lon, 'LineWidth',2)
hold on 
geoplot(latOver10, lonOver10, 'r', 'LineWidth', 2)
hold on
geoplot(llaG(1),llaG(2), '^', 'MarkerFaceColor','k', 'MarkerSize', 10)
geobasemap topographic;



