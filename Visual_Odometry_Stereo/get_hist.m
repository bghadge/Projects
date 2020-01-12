%%call
features = get_features(rgb2gray(imread('test_image.png')));
seed = 0;
n_c = 100; 
codewords = get_codewords(features, n_c, seed);
h = get_hist(codewords, features)

%%
function h = get_hist(codewords, features)
num_codewords = size(codewords, 1);
%% knnsearch and histograms
[idx,D] = knnsearch(codewords,features); 
[a,edges] = histcounts(idx,num_codewords);
h = (1/size(features,1))*a;

% `h` must be a row vector
assert(isequal(size(h), [1, num_codewords]))

% `h` must be normalized
assert((sum(h) - 1)^2 < eps)
end