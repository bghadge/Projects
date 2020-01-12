%% use this to run the code separately
w_radius = 4;
max_d = 64;
min_d = 0;
I1 = imread('t0_left.png');
I2 = imread('t0_right.png');
[D]=genDisparityMap(I1, I2, min_d, max_d, w_radius);
%%
function [D] = genDisparityMap(I1, I2, min_d, max_d, w_radius)
% INPUT
%   I1 the left stereo image
%   I2 the right stereo image
%   min_d minimum disparity
%   max_d maximum disparity
%   w_radius the radius of the window to do the AD aggeration
%
% OUTPUT
%   D disparity values

if nargin < 5, w_radius = 4; end % 9x9 window
if nargin < 4, max_d = 64; end
if nargin < 3, min_d = 0; end

% Grayscale Images are sufficient for stereo matching
% The Green channel is actually a good approximation of the grayscale, we
% could instad do I1 = I1(:,:,2);
if size(I1, 3) > 1, I1 = rgb2gray(I1); end
if size(I2, 3) > 1, I2 = rgb2gray(I2); end

% convert to double/single
I1 = double(I1);
I2 = double(I2);

%% Calculate SAD values for each pixel in the Left Image.
%% kernel
kernel = ones(9);

%%
% the range of disparity values from min_d to max_d inclusive
d_vals = min_d:max_d;
SAD = repmat(1000 * ones(size(I1,1), size(I1,2)), [1,1,max_d-min_d + 1]);
for i = min_d:max_d
    I2_shift = imtranslate(I2, [i,0]);
    abs_diff = abs(I1 - I2_shift);
    SAD(:,:,i+1) = conv2(abs_diff, kernel, 'same');
end
%% D is the Disparity Matrix and is the same size as that of the Images.
[~, D] = min(SAD, [], 3);
D = D + min_d -1;

%% Visualize disparity map
figure;
imagesc(D, [0, max_d]);
colormap(gray);
colorbar;
axis image;
end
