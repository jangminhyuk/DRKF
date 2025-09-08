function [params_eps, Sigma] = data(horizon,D_const,B_const,setting)
    %%%% Generate and fit some data
    % X_data is a matrix (horizon x samples), i.e., each column corresponds
    % to a trajectory.
    % params_eps is a (horizon x 2) matrix, where params_eps(i,1) = mu, params_eps(i,2) = sigma^2 
    % and eps_i follows N(mu,sigma^2)
    % Sigma is the Sample Covariance of the Y data, as needed in the FW
    % algorithm.
    samples = 20;
    X_data = zeros(horizon,1); 
    Y_data = zeros(horizon,samples);
    eps_data = zeros(horizon,samples);
    params_eps = zeros(horizon,2);
    for i = 1:samples
        X_data = 0*X_data;
        for j = 1:horizon
            if j == 1
                val = 1;
                X_data(j,1) = normrnd(0,1);
            else
                val = sum(X_data(1:j-1,1))*normrnd(0,1);%history_function(X_data(1:j-1,1),Y_data(1:j-1,i),hist_id);
                X_data(j,1) = D_const*X_data(j-1,1) + normrnd(0,1);
            end
            if setting == 1
                eps_data(j,i) = val*unifrnd(-1,1);
                Y_data(j,i) = B_const*X_data(j,1) + eps_data(j,i);
            else
                eps_data(j,i) = unifrnd(-1,1);  
                Y_data(j,i) = B_const*X_data(j,1) + val*eps_data(j,i);
            end
        end
    end
    %Compute sample covariance
    for j = 1:horizon
        params_eps(j,1) = 0;
        params_eps(j,2) = sum(eps_data(j,:).^2)/samples;
    end

    Sigma = diag(params_eps(:,2));
end
