function T_final = icp(p_t, p_s, use_naive, T_init)

m = size(p_t, 1);

if nargin < 4
    T_init = eye(m + 1);
    if nargin < 3
        use_naive = true;
    end
end

delta = inf;
T_final = T_init;
while delta > sqrt(eps)
    p1 = T_final(1:m, 1:m) * p_s + T_final(1:m, m + 1);

    %% Knn Search:
    [Idx,d] = knnsearch(p_t',p1');
    p2 = p_t(:,Idx);

    if use_naive
        weight = ones(size(p1, 2), 1);
    else

        %% weighted ICP
        std_dev = std(d);
        
        weight = exp(-((d.*d)/std_dev^2));

    end

    % incremental transformation
    [delta_R, delta_t] = rigid_fit(p1, p2, weight);

    %% update T_final with `delta_R` and `delta_t`
    T_final = [delta_R delta_t; 0 0 0 1] * T_final; 
    delta = norm([delta_R - eye(m), delta_t]);
end

end
