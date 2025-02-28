%% call
% Load the Images
I0_left = imread('t0_left.png');
I0_right = imread('t0_right.png');

% Load the Left and Right Intrinisc Camera Calibration Matrices
calib = load('IntrinsicMatrixLeftRight.mat');
P_left = calib.P{1};
P_right = calib.P{2};

% Detect Feature Points
[p0_left, p0_right, f0_left, ~] = detectFeaturePoints(I0_left, I0_right);

% Display Matched features
showMatchedFeatures(I0_left, I0_right, p0_left, p0_right);

% points in pixel coordinates (2D)
x0_left = double(p0_left.Location.');
x0_right = double(p0_right.Location.'); 

% Perform traingulation on the first feature location
X= linear_triangulation(x0_left(:, 1), x0_right(:, 1), P_left, P_right)

%% function def
function X = linear_triangulation(x_left, x_right, P_left, P_right)
%% perform linear triangulation to get 3D points from corresponding 2D points
%
% Notation:
%   2D points (lower case) are in pixel coordinate: x = [x1; x2]
%   3D points (upper case) are in world coordinate: X = [X1; X2; X3]

%% trainagulation matrix
A = [x_left(1,1)*P_left(3,:) - P_left(1,:);
     x_left(2,1)*P_left(3,:) - P_left(2,:);
     x_right(1,1)*P_right(3,:) - P_right(1,:);
     x_right(2,1)*P_right(3,:) - P_right(2,:)];

[U S V] = svd(A);

[m,i] = min(diag(S));
X = V((1:3),i)/V(4,i);

end
