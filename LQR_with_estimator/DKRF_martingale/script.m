setting = 1; % '1' for E1, '2' for E2
% deltas for T=5/E1
deltas = [0.00001 0.01 0.1 0.5 1 1.5 2 2.5 3 3.5 4 5 6 7 9 11.5];
horizon = 5;
output(deltas,setting,horizon);
% deltas for T=10/E1
deltas = [0.00001 1 3 5 7 10 13 15 17 20 25 30 35 40 45 50 60 63];
horizon = 10;
output(deltas,setting,horizon);

setting = 2;
% deltas for T=5/E2
deltas = [0.00001 0.5 1 3 5 7 9 11 13 15 17 17.9 21];
horizon = 5;
output(deltas,setting,horizon);
% deltas for T=10/E2
deltas = [0.00001 20 40 60 80 100 120 140 380];
horizon = 10;
output(deltas,setting,horizon);