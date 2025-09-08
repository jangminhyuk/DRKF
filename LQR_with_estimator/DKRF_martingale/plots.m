function plots(error,deltas,horizon)
    norm_full = error(1,1);
    norm_last = error(4,1);
    first = 1;
    last = length(deltas); 
    % Horizon MSE
    figure()
    hold on
    plot(deltas(first:last)/norm_full,error(1,first:last),'-o','DisplayName','MaDRE (ours)','LineWidth',1.2);
    plot(deltas(first:last)/norm_full,error(2,first:last),'-*','DisplayName','WMMSE-Conditional','LineWidth',1.2);
    plot(deltas(first:last)/norm_full,error(3,first:last),'-+','DisplayName','WMMSE-Future','LineWidth',1.2);
    title(['Horizon MSE for Τ = ', num2str(horizon)])
    xlabel('normalized \delta') 
    ylabel('MSE') 
    h = legend('Location','northwest');
    set(h,'FontSize',16);
    set(gca,"FontSize",16);
    grid on
    hold off

    %Last-step MSE
    figure()
    hold on
    plot(deltas(first:last-1)/norm_last,(error(5,first:last-1)),'-o','DisplayName','MaDRE (ours)','LineWidth',1.2);
    plot(deltas(first:last-1)/norm_last,(error(4,first:last-1)),'-*','DisplayName','WMMSE','LineWidth',1.2); 
    title(['Last-step MSE for Τ = ', num2str(horizon)])
    xlabel('normalized \delta') 
    ylabel('MSE') 
    h = legend('Location','northwest');
    set(h,'FontSize',16);
    set(gca,"FontSize",16);
    grid on
    hold off
end