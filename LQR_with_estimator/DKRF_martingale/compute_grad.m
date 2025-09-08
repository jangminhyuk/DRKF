function grad = compute_grad(phi,c,alpha,horizon,var_eps,var_eta,hatpsi,A,lambda)
    grad = zeros(horizon,horizon);
    for n = 1:horizon
        for m = 1:n
            for i = 1:m
                p = lambda-c(i,1);
                x_d = c(i,1) + p;
                y = alpha(i,1);
                var_til = -y*var_eta(i,1)/p;
                grad(n,m) = grad(n,m) - 2*var_eta(i,1)*A(n,i)*hatpsi(m,i) + 2*var_til*phi(n,i)*hatpsi(m,i);
            end
            var_tileps = (var_eps(m,1)*x_d^2 + var_eta(m,1)*y^2)/p^2;
            grad(n,m) = grad(n,m) + 2*phi(n,m)*var_tileps - 2*var_til*A(n,m);
        end
    end
end
