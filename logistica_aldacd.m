function logistica_aldacd_repeat
%% REGRESIÓN LOGÍSTICA y ANÁLISIS DISCRIMINANTE (ALD/ACD)

load Xtrain.mat
load Ytrain.mat

% Normalización
X = zscore(Xtrain);
Y = double(Ytrain);

%% ========= Configuración de evaluación por repetición =========
n_trials = 10;
best = struct('logit', 0, 'ald', 0, 'acd', 0);
metrics_logit = []; metrics_ald = []; metrics_acd = [];

for trial = 1:n_trials
    rng(trial);
    cv = cvpartition(length(Y), 'Holdout', 0.4);
    pos_train = training(cv);
    pos_test = test(cv);

    x1 = X(pos_train,:);
    x2 = X(pos_test,:);
    y1 = Y(pos_train);
    y2 = Y(pos_test);

    %% === REGRESIÓN LOGÍSTICA ===
    modelo_logit = fitglm(x1, y1, 'Distribution', 'binomial');
    probs_logit = predict(modelo_logit, x2);
    label_logit = probs_logit > 0.5;
    [~, ~, ~, BAC_logit] = compute_metrics(label_logit, y2);
    metrics_logit(end+1) = BAC_logit;
    if BAC_logit > best.logit
        best.logit = BAC_logit;
        best.label_logit = label_logit;
        best.y2_logit = y2;
        best.trial_logit = trial;
    end

    %% === ALD ===
    mdl_ald = fitcdiscr(x1, y1, 'DiscrimType', 'linear');
    label_ald = predict(mdl_ald, x2);
    [~, ~, ~, BAC_ald] = compute_metrics(label_ald, y2);
    metrics_ald(end+1) = BAC_ald;
    if BAC_ald > best.ald
        best.ald = BAC_ald;
        best.label_ald = label_ald;
        best.y2_ald = y2;
        best.trial_ald = trial;
    end

    %% === ACD ===
    mdl_acd = fitcdiscr(x1, y1, 'DiscrimType', 'quadratic');
    label_acd = predict(mdl_acd, x2);
    [~, ~, ~, BAC_acd] = compute_metrics(label_acd, y2);
    metrics_acd(end+1) = BAC_acd;
    if BAC_acd > best.acd
        best.acd = BAC_acd;
        best.label_acd = label_acd;
        best.y2_acd = y2;
        best.trial_acd = trial;
    end
end

%% ===================== RESULTADOS =====================
fprintf('\n>>> REGRESIÓN LOGÍSTICA <<<\n');
[SE_logit, SP_logit, ACC_logit, BAC_logit] = compute_metrics(best.label_logit, best.y2_logit);
fprintf('Trial = %d\n', best.trial_logit);
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_logit, SP_logit, ACC_logit, BAC_logit);

fprintf('\n>>> ALD (Discriminante Lineal) <<<\n');
[SE_ald, SP_ald, ACC_ald, BAC_ald] = compute_metrics(best.label_ald, best.y2_ald);
fprintf('Trial = %d\n', best.trial_ald);
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_ald, SP_ald, ACC_ald, BAC_ald);

fprintf('\n>>> ACD (Discriminante Cuadrático) <<<\n');
[SE_acd, SP_acd, ACC_acd, BAC_acd] = compute_metrics(best.label_acd, best.y2_acd);
fprintf('Trial = %d\n', best.trial_acd);
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_acd, SP_acd, ACC_acd, BAC_acd);

%% ===================== CONFUSION CHARTS =====================
figure;
subplot(1,3,1); confusionchart(double(best.y2_logit), double(best.label_logit)); title('Logística');
subplot(1,3,2); confusionchart(double(best.y2_ald), double(best.label_ald)); title('ALD');
subplot(1,3,3); confusionchart(double(best.y2_acd), double(best.label_acd)); title('ACD');

%% ===================== BOXPLOT DE BAC =====================
figure;
boxplot([metrics_logit(:), metrics_ald(:), metrics_acd(:)], ...
    'Labels', {'Logística', 'ALD', 'ACD'});
ylabel('BAC'); title('Variabilidad de BAC en 10 repeticiones');
grid on;
end
