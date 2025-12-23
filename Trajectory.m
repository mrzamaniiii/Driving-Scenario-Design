function [scenario, egoVehicle] = A()

scenario = drivingScenario('GeoReference', [45.477019 9.228644 0], ...
    'VerticalAxis', 'Y');

% Segment 1
roadCenters = [147.85972572493 -115.34048302817 -0.0027555102729906;
    147.89114819967 -121.5421544863  -0.0028715872062364;
    148.00121582408 -147.00457241343 -0.0034110336758175];
assert(all(isfinite(roadCenters(:))), 'roadCenters contains NaN/Inf or invalid numbers');
laneSpecification = lanespec([1 1]);
road(scenario, roadCenters, 'Lanes', laneSpecification, 'Name', '512347318');

% Segment 2
roadCenters = [147.82817222343 -104.63757984353 -1.0914588042197;
    147.85972572493 -115.34048302817 -0.0027555102729906];
assert(all(isfinite(roadCenters(:))), 'roadCenters contains NaN/Inf or invalid numbers');
laneSpecification = lanespec([1 1]);
road(scenario, roadCenters, 'Lanes', laneSpecification, 'Name', '512347318');

% Segment 3
roadCenters = [147.82817222343 -104.63757984353 -1.0914588042197;
    140.01687370368 -104.64887006654 -1.2614219170298;
    125.00417623142 -104.68252953643 -1.2270833233945;
    120.71930645029 -104.6937313394  -0.98880666492022;
    109.54580001863 -104.71617155447 -0.54624457132333];
%assert(all(isfinite(roadCenters(:))), 'roadCenters contains NaN/Inf or invalid numbers');
laneSpecification = lanespec([1 1]);
road(scenario, roadCenters, 'Lanes', laneSpecification, 'Name', '512877782-512877783-512877784');

% Ego vehicle
egoVehicle = vehicle(scenario, ...
    'ClassID', 1, ...
    'Position', [147.78 -145.08 0.01], ...
    'Mesh', driving.scenario.carMesh, ...
    'Name', 'Car');

waypoints = [147.78 -145.08 0.01;
    148    -136.15 0.01;
    147.41 -120.18 0.01;
    148.59 -107.73 -0.96;
    141.78 -104.44 -1.24;
    130.28 -104.88 -1.24;
    112.63 -104.8  -0.63];

speed = [30;30;30;30;30;30;30];
trajectory(egoVehicle, waypoints, speed);

end
