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

%% Main ICP

% Weighted Centroids
p1_bar = (p1 * weight)/sum(weight);
p2_bar = (p2 * weight)/sum(weight);

%centered vectors
X = p1 - p1_bar;
Y = p2 - p2_bar;

%covariance matrix
W = diag(weight);
S = X * W * Y';

%SVD
[U,Sigma,V] = svd(S);

%Rotation
O_v = zeros((size(V,1)-1),1);
O_h = zeros(1,(size(V,1)-1));
h = det(V * U');
M = [eye((size(V,1)-1)) O_v; O_h h];

%[1 0 0;0 1 0;0 0 det(V * U')]

R = V * M * U'

% Translation 
t = p2_bar - R * p1_bar

end