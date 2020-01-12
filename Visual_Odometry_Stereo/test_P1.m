%% To test each sub-problem, use Run Section!

filename = 'test_image.png';
%% P1.1
image = rgb2gray(imread(filename));
features = get_features(image);
features_ref = get_features_ref(image);

%% P1.2
seed = 0;
n_c = 100;
codewords = get_codewords(features, n_c, seed);
codewords_ref = get_codewords_ref(features_ref, n_c, seed);

%% P1.3
h = get_hist(codewords, features);
h_ref = get_hist_ref(codewords_ref, features_ref);

%% P1.4
[idx, dist] = query('kitti_bow.mat');
[idx_ref, dist_ref] = query_ref('kitti_bow.mat');
