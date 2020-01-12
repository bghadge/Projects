%% To test each sub-problem, use Run Section!

%% Q2.1
I0_left = imread('t0_left.png');
I0_right = imread('t0_right.png');

% !!!Your code runs here.
[matchedPoints_left0, matchedPoints_right0, matchedFeatures_left0, matchedFeatures_right0] = detectFeaturePoints(I0_left, I0_right);

% visualize
figure
showMatchedFeatures(I0_left,I0_right,matchedPoints_left0,matchedPoints_right0,'method','blend')
legend('Matched points from Left','Matched points from Right');

%% Q2.1 reference
[matchedPoints_left0_ref, matchedPoints_right0_ref, matchedFeatures_left0_ref, matchedFeatures_right0_ref] = detectFeaturePoints_ref(I0_left, I0_right);
figure
showMatchedFeatures(I0_left,I0_right,matchedPoints_left0_ref,matchedPoints_right0_ref,'method','blend')
legend('Matched points from Left','Matched points from Right');
title('Feature Match Reference')

%% Q2.2
% points in pixel coordinates (2D)
x0_left = double(matchedPoints_left0.Location.');
x0_right = double(matchedPoints_right0.Location.');

% Load the Left and Right Intrinisc Camera Calibration Matrices
calib = load('IntrinsicMatrixLeftRight.mat');
P_left = calib.P{1};
P_right = calib.P{2};

% Your code runs here.
% Perform traingulation on the first feature location
X0_1= linear_triangulation(x0_left(:, 1), x0_right(:, 1), P_left, P_right);

%% Q2.2 reference
x0_left_ref = double(matchedPoints_left0_ref.Location.');
x0_right_ref = double(matchedPoints_right0_ref.Location.');
X0_1_ref= linear_triangulation_ref(x0_left_ref(:, 1), x0_right_ref(:, 1), P_left, P_right);

%% Q2.3
I1_left = imread('t1_left.png');
I1_right = imread('t1_right.png');
[matchedPoints_left1, matchedPoints_right1, matchedFeatures_left1, matchedFeatures_right1] = detectFeaturePoints(I1_left, I1_right);

% points in pixel coordinates (2D)
x1_left = double(matchedPoints_left1.Location.');
x1_right = double(matchedPoints_right1.Location.');

% generate 3D points
X0 = zeros(3,size(x0_left,2));
for i=1:size(x0_left,2)
    X0(:,i) = linear_triangulation(x0_left(:, i), x0_right(:, i), P_left, P_right);
end
X1 = zeros(3,size(x1_left,2));
for i=1:size(x0_left,2)
    X1(:,i) = linear_triangulation(x1_left(:, i), x1_right(:, i), P_left, P_right);
end

% find data assosciation between two timestamps
idx_match = matchFeatures(matchedFeatures_left0,matchedFeatures_left1);

% !!!Your code runs here.
[T, idx_inliers] = visual_odom_ransac(...
    X0(:, idx_match(:, 1)), X1(:, idx_match(:, 2)), ...
    x0_left(:, idx_match(:, 1)), x1_left(:, idx_match(:, 2)), ...
    P_left, seed);

% show 3D points (x = camera right, y = camera down, z = camera forward)
figure
set(gcf, 'position', [200, 200, 600, 800])
pcshowpair(pointCloud(X0'), pctransform(pointCloud(X1'),affine3d(T')),'MarkerSize',10)
axis([-25, 25, -10, 10, 0, 50])

%% Q2.3 reference
[matchedPoints_left1_ref, matchedPoints_right1_ref, matchedFeatures_left1_ref, matchedFeatures_right1_ref] = detectFeaturePoints_ref(I1_left, I1_right);

% points in pixel coordinates (2D)
x1_left_ref = double(matchedPoints_left1_ref.Location.');
x1_right_ref = double(matchedPoints_right1_ref.Location.');

% generate 3D points
X0_ref = zeros(3,size(x0_left_ref,2));
for i=1:size(x0_left_ref,2)
    X0_ref(:,i) = linear_triangulation_ref(x0_left_ref(:, i), x0_right_ref(:, i), P_left, P_right);
end
X1_ref = zeros(3,size(x1_left_ref,2));
for i=1:size(x0_left_ref,2)
    X1_ref(:,i) = linear_triangulation_ref(x1_left_ref(:, i), x1_right_ref(:, i), P_left, P_right);
end
% find data assosciation between two timestamps
idx_match_ref = matchFeatures(matchedFeatures_left0_ref,matchedFeatures_left1_ref);
[T_ref, idx_inliers_ref] = visual_odom_ransac_ref(...
    X0_ref(:, idx_match_ref(:, 1)), X1_ref(:, idx_match_ref(:, 2)), ...
    x0_left_ref(:, idx_match_ref(:, 1)), x1_left_ref(:, idx_match_ref(:, 2)), ...
    P_left, seed);
% show 3D points (x = camera right, y = camera down, z = camera forward)
figure
set(gcf, 'position', [200, 200, 600, 800])
pcshowpair(pointCloud(X0_ref'), pctransform(pointCloud(X1_ref'),affine3d(T_ref')),'MarkerSize',10)
axis([-25, 25, -10, 10, 0, 50])
title('Reference Point Match')
