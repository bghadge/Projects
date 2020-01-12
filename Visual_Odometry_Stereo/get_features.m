%% run in section
image = rgb2gray(imread('test_image.png'))
features = get_features(image);

%% function def
function features = get_features(image)
points = detectSURFFeatures(image);
[features,validPoints] = extractFeatures(image,points);
features = double(features);
end