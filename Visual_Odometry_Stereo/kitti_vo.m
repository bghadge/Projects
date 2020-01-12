clc
%% load stereo images for two frames (i.e. at t0 and t1)
I0_left = imread('t0_left.png');
I0_right = imread('t0_right.png');

I1_left = imread('t1_left.png');
I1_right = imread('t1_right.png');

%% load camera matrices
calib = load('IntrinsicMatrixLeftRight.mat');
P_left = calib.P{1};
P_right = calib.P{2};

%%
[p0_left, p0_right, f0_left, ~] = detectFeaturePoints(I0_left, I0_right);
[p1_left, p1_right, f1_left, ~] = detectFeaturePoints(I1_left, I1_right);

%% show stereo images
figure(1)
set(gcf, 'position', [100, 100, 800, 600])

subplot(2, 1, 1)
showMatchedFeatures(I0_left, I0_right, p0_left, p0_right)
title('t0')

subplot(2, 1, 2)
showMatchedFeatures(I1_left, I1_right, p1_left, p1_right)
title('t1')

%% points in pixel coordinates (2D)
x0_left = double(p0_left.Location.');
x0_right = double(p0_right.Location.');

x1_left = double(p1_left.Location.');
x1_right = double(p1_right.Location.');

%% triangulate points to 3D
n0 = size(x0_left, 2);
X0 = nan(3, n0);
for k = 1:n0
    X0(:, k) = linear_triangulation(x0_left(:, k), x0_right(:, k), P_left, P_right);
end

n1 = size(x1_left, 2);
X1 = nan(3, n1);
for k = 1:n1
    X1(:, k) = linear_triangulation(x1_left(:, k), x1_right(:, k), P_left, P_right);
end

%% show 3D points (x = camera right, y = camera down, z = camera forward)
figure(2)
set(gcf, 'position', [200, 200, 600, 800])

pcshow(X0.', 'r'); hold on
pcshow(X1.', 'b'); hold off
axis([-25, 25, -10, 10, 0, 50])

%% RANSAC
idx_match = matchFeatures(f0_left, f1_left);
[T, idx_inliers] = visual_odom_ransac(...
    X0(:, idx_match(:, 1)), X1(:, idx_match(:, 2)), ...
    x0_left(:, idx_match(:, 1)), x1_left(:, idx_match(:, 2)), ...
    P_left);

save('variables.mat', 'P_left', 'P_right', 'idx_match', ...
    'X0', 'x0_left', 'x0_right', ...
    'X1', 'x1_left', 'x1_right')

% prior information to this problem:
%   1. less than 10% of points are outliers
%   2. translation in z is at least 0.5 meters
while mean(idx_inliers) < 0.9 || T(3, 4) < 0.5
    disp('Retrying ...')
    [T, idx_inliers] = visual_odom_ransac(...
    X0(:, idx_match(:, 1)), X1(:, idx_match(:, 2)), ...
    x0_left(:, idx_match(:, 1)), x1_left(:, idx_match(:, 2)), ...
    P_left);
end

fprintf('Done.\nT =\n')
disp(T)

disp(num2str(mean(idx_inliers) * 100, 'Percentage of inliers = %.2f'))
