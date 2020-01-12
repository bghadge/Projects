function [matchedPoints1, matchedPoints2, matchedFeatures1, matchedFeatures2] = detectFeaturePoints(I1, I2)
if(size(I1,3) == 3)
    I1 = rgb2gray(I1);
end

if(size(I2,3) == 3)
    I2 = rgb2gray(I2);
end

% detect SURF points
points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);

%% matchFeatures
[features1,valid1] = extractFeatures(I1,points1);
[features2,valid2] = extractFeatures(I2,points2);

indices = matchFeatures(features1,features2);

matchedPoints1 = valid1(indices(:,1));
matchedPoints2 = valid2(indices(:,2));
matchedFeatures1 = features1(indices(:,1),:);
matchedFeatures2 = features2(indices(:,2),:);

end