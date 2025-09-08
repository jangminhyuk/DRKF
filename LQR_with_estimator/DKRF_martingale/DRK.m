function phi = DRK(delta,horizon,var_eps,var_eta,D_const,B_const)

    % Kalman constants
    D = D_const*ones(horizon,1);
    B = B_const*ones(horizon,1);
    
    %Compute psi and hatpsi
    psi = ones(horizon,horizon);
    hatpsi = ones(horizon,horizon);
    for n = 1:horizon
        for i = 1:horizon
            for j = (i+1):n
                psi(n,i) = psi(n,i)*D(j,1);
            end
            hatpsi(n,i) = B(n,1)*psi(n,i);
        end
    end
    
    % Run subgradient descent
    A = zeros(horizon,horizon);
    c = zeros(horizon,1);
    kappa = zeros(horizon,1);
    alpha = zeros(horizon,1);
    phi = ones(horizon,horizon);
    flag = 1;
    t = 1;
    while flag == 1
        %Compute A
        for n = 1:horizon
            for i = 1:n
                A(n,i) = psi(n,i);
                for j = i:n
                    A(n,i) =  A(n,i) - phi(n,j)*hatpsi(j,i);
                end
            end
        end
    
        % Compute c & alpha
        alpha = 0*alpha;
        c = 0*c;
        kappa = 0*kappa;
        for i = 1:horizon
            for n = i:horizon
                c(i,1) = c(i,1) + phi(n,i)^2;
                alpha(i,1) = alpha(i,1) + A(n,i)*phi(n,i);
                kappa(i,1) = kappa(i,1) + A(n,i)^2;
            end
        end
        
        %Compute lambda and gradient
        lambda = compute_lambda(delta,c,alpha,horizon,var_eps,var_eta);
        grad = compute_grad(phi,c,alpha,horizon,var_eps,var_eta,hatpsi,A,lambda);
        %Update step
        step = 1/t;
        phi = phi - step*grad/norm(grad);
        t = t+1;
        if norm(grad,1) < 0.000001
            flag = 0;
        end
        if (t > 2000)
            break;
        end
    end
end

