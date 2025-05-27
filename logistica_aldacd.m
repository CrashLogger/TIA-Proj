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

%% ===================== ÁRBOL DE CLASIFICACIÓN =====================

% Árbol inicial
tree = fitctree(x1, y1, 'SplitCriterion', 'gdi');

% Predecimos en test
label_tree = predict(tree, x2);
[SE_tree, SP_tree, ACC_tree, BAC_tree] = compute_metrics(label_tree, y2);

% Visualización inicial
view(tree, 'Mode', 'graph');

%% ===================== PODA CON VALIDACIÓN CRUZADA =====================

alpha_grid = tree.PruneAlpha;
rng(2);
k = 10;
c = cvpartition(length(y1), 'KFold', k);
CV_error = [];

for aa = 1:k
    idx_tr = training(c, aa);
    idx_te = test(c, aa);
    
    Xtr = x1(idx_tr,:);
    Xte = x1(idx_te,:);
    Ytr = y1(idx_tr);
    Yte = y1(idx_te);

    tree_cv = fitctree(Xtr, Ytr, 'SplitCriterion', 'gdi');

    for bb = 1:length(alpha_grid)-1
        podado = prune(tree_cv, 'Alpha', alpha_grid(bb));
        pred = predict(podado, Xte);
        CV_error(aa,bb) = 100 * (1 - sum(pred == Yte)/length(Yte));
    end
end

% Selección del mejor alpha
[~, pos] = min(mean(CV_error));
alpha_opt = alpha_grid(pos);

% Árbol podado definitivo
tree_pruned = prune(tree, 'Alpha', alpha_opt);

% Evaluar árbol podado
label_pruned = predict(tree_pruned, x2);
[SE_treeP, SP_treeP, ACC_treeP, BAC_treeP] = compute_metrics(label_pruned, y2);

% Visualización árbol podado
view(tree_pruned, 'Mode', 'graph');

%% ===================== RESULTADOS =====================

fprintf('\n>>> REGRESIÓN LOGÍSTICA <<<\n');
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_logit, SP_logit, ACC_logit, BAC_logit);

fprintf('\n>>> ÁRBOL DE CLASIFICACIÓN SIN PODAR <<<\n');
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_tree, SP_tree, ACC_tree, BAC_tree);

fprintf('\n>>> ÁRBOL DE CLASIFICACIÓN PODADO <<<\n');
fprintf('Alpha = %.4f | Nodos terminales = %d\n', alpha_opt, sum(~tree_pruned.IsBranchNode));
fprintf('SE = %.4f | SP = %.4f | ACC = %.4f | BAC = %.4f\n', SE_treeP, SP_treeP, ACC_treeP, BAC_treeP);

%% ===================== CONFUSION CHARTS =====================

figure;
subplot(1,3,1); confusionchart(double(y2), double(label_logit)); title('Logística');
subplot(1,3,2); confusionchart(double(y2), double(label_tree)); title('Árbol sin podar');
subplot(1,3,3); confusionchart(double(y2), double(label_pruned)); title('Árbol podado');
pause; close;

%% ===================== ERRORBAR PODA =====================
figure;
errorbar(alpha_grid(1:end-1), mean(CV_error), std(CV_error));
hold on; plot(alpha_opt, mean(CV_error(:,pos)), 'ro');
xlabel('Alpha'); ylabel('Error de validación cruzada (%)');
title('Selección de α - Poda del árbol');
grid on;
pause; close;

end
