%% RIDGE y LASSO sobre Xtrain / Ytrain
function ridge_regression_Lasso

load Xtrain.mat
load Ytrain.mat

% Normalizar
Xtrain = zscore(Xtrain);

% Dividir en train/test (60/40)
rng(1);
cv = cvpartition(length(Ytrain), 'Holdout', 0.4);
pos_train = training(cv);
pos_test = test(cv);

x1 = Xtrain(pos_train,:);
x2 = Xtrain(pos_test,:);
y1 = Ytrain(pos_train);
y2 = Ytrain(pos_test);

fprintf('Tamaño entrenamiento: %d | test: %d\n', sum(pos_train), sum(pos_test));

%% ================= RIDGE REGRESSION =================

% Visualización - Ridge
lambda_grid = logspace(-2, 2, 100);
coefs_ridge = [];

for i = 1:length(lambda_grid)
    B = ridge(y1, x1, lambda_grid(i), 0);
    coefs_ridge(:,i) = B(2:end); % Se quita el intercepto --> grafica solo variables
end

% Visualización de coeficientes
figure;
semilogx(lambda_grid, coefs_ridge');
xlabel('Lambda');
ylabel('Coeficientes');
title('RIDGE REGRESSION');
grid on;

%% ================= LASSO =================

% Visualización - Lasso
[B, FitInfo] = lassoglm(x1, y1, 'binomial', 'Lambda', lambda_grid, 'CV', 10, 'Standardize', false);

% Visualización de coeficientes
figure;
lassoPlot(B, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');
title('LASSO');


%% ========== COMPARACIÓN MSE (VALIDACIÓN CRUZADA) ==========

rng(2); %Semilla para CV
k = 10;
c = cvpartition(length(y1), 'KFold', k);
CV_MSE = zeros(k, length(lambda_grid));
CV_MSE_LASSO = zeros(k, length(lambda_grid));

for fold = 1:k
    idx_train = training(c, fold);
    idx_test = test(c, fold);

    Xt = x1(idx_train,:);
    Yt = y1(idx_train);
    Xt_test = x1(idx_test,:);
    Yt_test = y1(idx_test);

    for j = 1:length(lambda_grid)
        % Ridge
        B_ridge = ridge(Yt, Xt, lambda_grid(j), 0);
        Yhat_ridge = [ones(size(Xt_test,1),1) Xt_test] * B_ridge;
        CV_MSE(fold,j) = mean((Yt_test - Yhat_ridge).^2);

        % Lasso
        [B_las, Fit] = lassoglm(Xt, Yt, 'binomial', 'Lambda', lambda_grid(j), 'Standardize', false);
        scores = Xt_test * B_las + Fit.Intercept;
        probs = 1 ./ (1 + exp(-scores));
        CV_MSE_LASSO(fold,j) = mean((Yt_test - probs).^2);
    end
end

% Visualización de MSE CV: Errores
[~, pos_ridge] = min(mean(CV_MSE));
[~, pos_lasso] = min(mean(CV_MSE_LASSO));

figure;
subplot(2,1,1);
semilogx(lambda_grid, mean(CV_MSE));
hold on; plot(lambda_grid(pos_ridge), mean(CV_MSE(:,pos_ridge)), 'ro'); hold off;
title('Ridge - Error de validación cruzada');
xlabel('Lambda'); ylabel('MSE');

subplot(2,1,2);
semilogx(lambda_grid, mean(CV_MSE_LASSO));
hold on; plot(lambda_grid(pos_lasso), mean(CV_MSE_LASSO(:,pos_lasso)), 'ro'); hold off;
title('Lasso - Error de validación cruzada');
xlabel('Lambda'); ylabel('MSE');

%% ========== TEST FINAL CON LOS MEJORES LAMBDA ==========

% Lambda óptimos
lambda_ridge = lambda_grid(pos_ridge);
lambda_lasso = lambda_grid(pos_lasso);

% Ridge
B_ridge = ridge(y1, x1, lambda_ridge, 0);
Yhat_ridge = [ones(size(x2,1),1) x2] * B_ridge;
label_ridge = round(Yhat_ridge);
MSE_test_ridge = mean((y2 - Yhat_ridge).^2);
[SE_ridge, SP_ridge, ACC_ridge, BAC_ridge] = compute_metrics(label_ridge, y2);

% Lasso
[B_lasso, Fit] = lassoglm(x1, y1, 'binomial', 'Lambda', lambda_lasso, 'Standardize', false);
scores = x2 * B_lasso + Fit.Intercept;
probs = 1 ./ (1 + exp(-scores));
label_lasso = probs > 0.5;
MSE_test_lasso = mean((y2 - probs).^2);
[SE_lasso, SP_lasso, ACC_lasso, BAC_lasso] = compute_metrics(label_lasso, y2);


%% ================== Resultados ==================


fprintf('\n>> RIDGE:\n');
fprintf('Lambda = %.4f | MSE = %.4f\n', lambda_ridge, MSE_test_ridge);
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_ridge, SP_ridge, ACC_ridge, BAC_ridge);

fprintf('\n>> LASSO:\n');
fprintf('Lambda = %.4f | MSE = %.4f\n', lambda_lasso, MSE_test_lasso);
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_lasso, SP_lasso, ACC_lasso, BAC_lasso);

% Confusion charts
figure;
subplot(1,2,1);
confusionchart(double(y2), double(label_ridge));
title('Ridge');

subplot(1,2,2);
confusionchart(double(y2), double(label_lasso));
title('Lasso');
