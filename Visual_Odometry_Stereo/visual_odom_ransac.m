%% call
load('variables.mat')

%% show 3D points (x = camera right, y = camera down, z = camera forward)
figure(2)
set(gcf, 'position', [200, 200, 600, 800])
pcshow(X0.', 'r'); hold on
pcshow(X1.', 'b'); hold off
axis([-25, 25, -10, 10, 0, 50])

%% RANSAC
[T, idx_inliers] = visual_odom_ransac(...
    X0(:, idx_match(:, 1)), X1(:, idx_match(:, 2)), ...
    x0_left(:, idx_match(:, 1)), x1_left(:, idx_match(:, 2)), ...
    P_left,0);

fprintf('Done.\nT =\n')
disp(T)

disp(num2str(mean(idx_inliers) * 100, 'Percentage of inliers = %.2f'))

%% function def
function [T, idx_inliers] = visual_odom_ransac(X0, X1, x0, x1, P_left, seed)
% Visual Odometry with RANSAC:
% estimate the relative motion (R, t) between two frames (i.e. t0 and t1)
% Capital X represents 3D points w.r.t camera frame
% Lower case x represents 2D pixel locations in image

assert(isequal(size(X0), size(X1)))
assert(isequal(size(x0), size(x1)))
assert(isequal(size(X0, 2), size(x0, 2)))

if nargin == 6
    rng(seed)
end

%% hyperparameters for RANSAC
% smallest number of points required to fit the model
s = 3;
% Number of iterations required to find transformation matrix using RANSAC
num_iter = 500;
% Threshold of the reprojection error
d_thresh = 12.5;

%%
num_points = size(X0, 2); 
X0 = [X0; ones(1, num_points)];     %making homogenous = 4 X nump
X1 = [X1; ones(1, num_points)];     %making homogenous = 4 X nump

% reprojection error of all points at each iteration
d = nan(num_iter, num_points);      %num_iter X nump

for k = 1:num_iter
    % draw samples from the two clouds
    idx_sample = randsample(num_points, s);     %3 X 1
    X0_sample = X0(:, idx_sample);              %4 X 3
    X1_sample = X1(:, idx_sample);              %4 X 3

    % Performing Rigid Fit
    %% Write your code here:
    [R, t] = rigid_fit(X1_sample(1:3,:),X0_sample(1:3,:));  %R 3X3 and t 3X1
    T = [R, t;0 0 0 1];                                     %4X4
    
    % Transform point clouds X0 and X1 to frame 1 and 0 respectively
    %% Write your code here:
    X0_in_1 = (inv(T))*X0;      %4Xnump
    X1_in_0 = T*X1;             %4Xnump

    % Projecting X1_in_0 and X0_in_1 to the respective 2D image planes
    %% Write your code here:
    x0_hat = P_left*X0_in_1;                    %3Xnump
    x0_hat = x0_hat(1:2,:) ./ x0_hat(3,:);      %non-homogenous 2Xnump
    x1_hat = P_left*X1_in_0;                    %3Xnump
    x1_hat = x1_hat(1:2,:) ./ x1_hat(3,:);      %non-homogenous 2Xnump

    % Calculating reprojection error d(:, k)
    %% Write your code here:
     d(k, :) = (vecnorm(x0 - x1_hat)).^2 + (vecnorm(x1 - x0_hat)).^2;       %1Xnump after loops 500Xnump

end

%% obtain indices of inliers corresponding to the iteration with the most number of inliers

[m,i] = max(sum((d<d_thresh),2));       %m=max value and i=index 
idx_inliers = (d(i,:)<d_thresh);    %row vector

%% Fit using all inliers

[R_f, t_f] = rigid_fit(X1(1:3,idx_inliers(1,:)),X0(1:3,idx_inliers(1,:)));
T = [R_f t_f;0 0 0 1];

end

function [R, t] = rigid_fit(p1, p2, weight)
%% Fit a rigid body transformation [R, t] by solving
%
%       \min    sum(w(i)* norm(R * p1(:, i) + t - p2(:, i) )^2)
%
%   Reference: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

assert(isequal(size(p1), size(p2)))
assert(size(p1, 1) <= size(p1, 2))

if nargin < 3
    weight = ones(size(p1, 2), 1);
end

assert(all(weight >= 0))

%% reshape and normalize
weight = reshape(weight, [], 1);
weight = weight / sum(weight);

%% Reference solution:
mu1 = p1 * weight;
mu2 = p2 * weight;

[U, ~, V] = svd((p1 - mu1) .* weight' * (p2 - mu2)');

D = eye(size(p1, 1));
if det(U) * det(V) < 0
    D(end, end) = -D(end, end);
end

R = V * D * U';
t = mu2 - R * mu1;
end
