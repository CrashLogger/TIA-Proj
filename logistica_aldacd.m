function logistica_aldacd
%% REGRESIÓN LOGÍSTICA y ÁRBOL DE CLASIFICACIÓN (ALD/ACD)

load Xtrain.mat
load Ytrain.mat

% Normalización
X = zscore(Xtrain);
Y = double(Ytrain);

% División Train/Test (60/40)
rng(1);
cv = cvpartition(length(Y), 'Holdout', 0.4);
pos_train = training(cv);
pos_test = test(cv);

x1 = X(pos_train,:);
x2 = X(pos_test,:);
y1 = Y(pos_train);
y2 = Y(pos_test);

fprintf('Tamaño entrenamiento: %d | Test: %d\n', sum(pos_train), sum(pos_test));

%% ===================== REGRESIÓN LOGÍSTICA =====================

% Entrenar modelo
modelo_logit = fitglm(x1, y1, 'Distribution', 'binomial');

% Predicciones
probs_logit = predict(modelo_logit, x2);
label_logit = probs_logit > 0.5;

% Evaluación
[SE_logit, SP_logit, ACC_logit, BAC_logit] = compute_metrics(label_logit, y2);

%% ===================== ALD (Análisis Lineal Discriminante) =====================

mdl_ald = fitcdiscr(x1, y1, 'DiscrimType', 'linear');
label_ald = predict(mdl_ald, x2);
[SE_ald, SP_ald, ACC_ald, BAC_ald] = compute_metrics(label_ald, y2);

%% ===================== ACD (Análisis Cuadrático Discriminante) =====================

mdl_acd = fitcdiscr(x1, y1, 'DiscrimType', 'quadratic');
label_acd = predict(mdl_acd, x2);
[SE_acd, SP_acd, ACC_acd, BAC_acd] = compute_metrics(label_acd, y2);

%% ===================== RESULTADOS =====================

fprintf('\n>>> REGRESIÓN LOGÍSTICA <<<\n');
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_logit, SP_logit, ACC_logit, BAC_logit);

fprintf('\n>>> ALD (Discriminante Lineal) <<<\n');
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_ald, SP_ald, ACC_ald, BAC_ald);

fprintf('\n>>> ACD (Discriminante Cuadrático) <<<\n');
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_acd, SP_acd, ACC_acd, BAC_acd);

%% ===================== CONFUSION CHARTS =====================

figure;
subplot(1,3,1); confusionchart(double(y2), double(label_logit)); title('Regresión Logística');
subplot(1,3,2); confusionchart(double(y2), double(label_ald)); title('ALD');
subplot(1,3,3); confusionchart(double(y2), double(label_acd)); title('ACD');
pause; close;

