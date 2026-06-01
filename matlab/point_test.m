% point_test.m
% Modified dari point.m Kilian untuk verifikasi dengan Python
% Load data dari CSV, export CI dan IRF point estimate ke CSV

global h p q

rand('state', 42);
randn('state', 42);

% Load test data
y = csvread('test_data_var.csv');   % (T x q)

p = 2;    % VAR lag order (sesuai DGP)
h = 12;   % impulse response horizon

[t, q] = size(y);
y = detrend(y, 0);

% Estimasi VAR
[A, SIGMA, U, V] = olsvarc(y, p);

% Bias correction jika stasioner
if ~any(abs(eig(A)) >= 1)
    [A] = asybc(A, SIGMA, t, p);
end

% Point estimate IRF
[IRF] = irfvar(A, SIGMA(1:q, 1:q), p);
IRF_mat = reshape(IRF, q^2, h+1)';   % (h+1) x q^2

% Bootstrap CI
[CI] = boot(A, U, y, V);

% CI: (2 x q^2*(h+1)) → reshape ke (h+1) x q^2 x 2
CI_lower = reshape(CI(1,:), q^2, h+1)';   % (h+1) x q^2
CI_upper = reshape(CI(2,:), q^2, h+1)';

% Export
csvwrite('irf_point_matlab.csv',  IRF_mat);
csvwrite('ci_lower_matlab.csv',   CI_lower);
csvwrite('ci_upper_matlab.csv',   CI_upper);

disp('Saved: irf_point_matlab.csv, ci_lower_matlab.csv, ci_upper_matlab.csv');
disp('IRF point estimate (first 5 horizons):');
disp(IRF_mat(1:5, :));
disp('CI lower (first 5 horizons):');
disp(CI_lower(1:5, :));
disp('CI upper (first 5 horizons):');
disp(CI_upper(1:5, :));