function lamda_value = compute_lambda(delta,c,alpha,horizon,var_eps,var_eta)
    g = -delta;
    for i = 1:horizon
        g = g + tf(var_eps(i,1)*c(i,1)^2 + var_eta(i,1)*alpha(i,1)^2,[1 -2*c(i,1) (c(i,1)^2)]);
    end
    G_min = zpk(minreal(g));    
    [Z,gain] = zero(G_min);    
    r = Z(imag(Z) == 0);
    lamda_value = max(r);
end
