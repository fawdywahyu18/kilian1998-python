% generate_test_data.m
% Generate VAR(2) bivariate test data dan export ke CSV

rand('state', 42);
randn('state', 42);

% True parameters
T   = 200;
q   = 2;
p   = 2;

A1 = [0.5, 0.1; 0.2, 0.4];
A2 = [0.1, 0.05; 0.05, 0.1];

SIGMA_true = [1.0, 0.3; 0.3, 1.0];
C = chol(SIGMA_true)';   % lower triangular

% Companion matrix (for stationarity check)
A_comp = [A1, A2; eye(q), zeros(q)];
disp('Max eigenvalue modulus (should be < 1):');
disp(max(abs(eig(A_comp))));

% Generate data
y = zeros(q, T);
y(:,1) = zeros(q,1);
y(:,2) = zeros(q,1);

for t = 3:T
    eps      = C * randn(q,1);
    y(:,t)   = A1*y(:,t-1) + A2*y(:,t-2) + eps;
end

% Export: rows = time, cols = variables
y_out = y';
csvwrite('test_data_var.csv', y_out);
disp('Saved: test_data_var.csv');
disp(size(y_out));