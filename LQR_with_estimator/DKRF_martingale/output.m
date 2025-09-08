function output(deltas,setting,horizon)
    rng(10);
    
    %Parameters of nominal
    D_const = 1;
    B_const = 1;
    % Fit data and compute the distribution of eps
    [params_eps, Sigma] = data(horizon,D_const,B_const,setting);
    % Set variance of etas
    var_eta = ones(horizon,1);
    var_eps = params_eps(:,2);
    
    num_del = size(deltas);
    num_del = num_del(1,2);
    mu_x = zeros(horizon,1);
    cov_x = zeros(horizon,horizon);   % Construct the covariance of X
    for i = 1:horizon
        for j = 1:horizon
            cov_x(i,j) = min(i,j);
        end
    end
    mu_w = zeros(horizon,1);
    cov_w = Sigma;
    delta_x = 0;
    H = B_const*eye(horizon);
    
    for i = 1:num_del    % Evaluate for each delta size
        delta = deltas(1,i);
        phi(:,:,i) = DRK(delta,horizon,var_eps,var_eta,D_const,B_const); % Compute our optimal phi's
        [A, b, obj, res, var_cov_x(:,:,i), var_cov_w(:,:,i)] = FrankWolfe(mu_x, cov_x, delta_x, mu_w, cov_w, sqrt(delta), H); % Compute FW optimal estimator 
        A_star(:,:,i) = A;
        b_star(:,:,i) = b;
        var_cov_y(:,:,i) = H*var_cov_x(:,:,i)*H' + var_cov_w(:,:,i);
    end
    error = evaluate(phi,A_star,b_star,num_del,horizon,D_const,B_const,var_cov_y);
    plots(error,deltas,horizon);
end