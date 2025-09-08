function error = evaluate(phi,A_star,b_star,num_del,horizon,D_const,B_const,cov_y)
    times = 1000;
    error = zeros(5,num_del);
    X = zeros(horizon,1);
    Y = zeros(horizon,1);
    for i = 1:num_del
        rng(1394);
        temp1 = 0;
        temp2 = 0;
        temp3 = 0;
        temp4 = 0;
        temp5 = 0;
        conditional = {};
        for m = 1:(horizon-1)
            temp_var = cov_y(1:m,1:m,i);
            temp_cov = cov_y((m+1):horizon,1:m,i);
            conditional{m} = temp_cov/temp_var;
        end
        for it = 1:times % Run #times and take the avg
            % Generate data
            X = 0*X;
            Y = 0*Y;
            for j = 1:horizon
                if j == 1
                    val = 1;
                    X(j,1) = normrnd(0,1);
                else
                    val = sum(X(1:j-1,1))*normrnd(0,1);
                    X(j,1) = D_const*X(j-1,1) + normrnd(0,1);
                end
                Y(j,1) = B_const*X(j,1) + val*unifrnd(-1,1);
            end
            % Test data
            err1 = 0;
            err2 = 0;
            for k = 1:horizon
                err1 = err1 + (X(k,1) - phi(k,1:k,i)*Y(1:k,1))^2;
                if k ~= horizon
                    err2 = err2 + ( X(k,1) - A_star(k,1:k,i)*Y(1:k,1) - A_star(k,(k+1):horizon,i)*conditional{k}*Y(1:k,1) )^2;
                else
                    err2 = err2 + (X(k,1) - A_star(k,:,i)*Y)^2;
                end
            end
            err3 = norm(X - A_star(:,:,i)*Y)^2;
            err4 = (X(k,1) - A_star(k,:,i)*Y - b_star(k,i))^2;
            err5 = (X(k,1) - phi(k,:,i)*Y)^2;
            % Total error from #times runs
            temp1 = temp1 + err1; 
            temp2 = temp2 + err2;    
            temp3 = temp3 + err3;
            temp4 = temp4 + err4;
            temp5 = temp5 + err5;
        end
        error(1,i) = temp1/times;
        error(2,i) = temp2/times;
        error(3,i) = temp3/times;
        error(4,i) = temp4/times;
        error(5,i) = temp5/times;
    end
end
