function trees
%TREES para el proyecto

load("Xtrain.mat")
load("Ytrain.mat")

close all;

% Se que es cualitativo y estoy usando MSE para los Cross-Validation.
% No se que me estaba pasando en la cabeza. Lo arreglaré pronto :tm:.

% Pese a que el archivo se llame [X/Y]train, voy a tratarlo como si fuera [X/Y] entero.

X = Xtrain;
Y = Ytrain;

%% Fijar una semilla para poder probar de manera consistente
%  Primero se hace un arbol de prueba tonto

rng(1);

% 50/50 de train y test
hpartition = cvpartition(size(X, 1), "HoldOut", 0.5);
pos_train = hpartition.training;
pos_test = hpartition.test;

tree = fitctree(X(pos_train, :), Y(pos_train));
alpha_grid = tree.PruneAlpha;

%% Evaluación del arbol inicial

view(tree, "Mode", "graph");
ypred = predict(tree, X(pos_test,:));
%MSE = mean((Y(pos_test)-ypred).^2)
[SE_orig,SP_orig,ACC_orig,BAC_orig] = compute_metrics(ypred, Y(pos_test));
fprintf('BAC del árbol de clasificación entero (nodos terminales=%d) = %4.2f \n\n',sum(~tree.IsBranchNode), BAC_orig);

%% Cross-Validation
%  Usamos K-Fold porque no hay motivo para no usarlo

rng(2)
k = 10;
c = cvpartition(sum(pos_train),'KFold',k);

X1 = X(pos_train,:);
Y1 = Y(pos_train);

CV_MSE=[];
for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    
    X_train = X1(pos_train_CV,:);
    X_test = X1(pos_test_CV,:);
    Y_train = Y1(pos_train_CV);
    Y_test = Y1(pos_test_CV);
    
    % Entrenamos árbol
    tree_train = fitctree(X_train, Y_train);
    
    % Para cada lambda, ajustamos y evaluamos
    for bb=1:length(alpha_grid)-1
        tree2 = prune(tree_train, "Alpha", alpha_grid(bb));
        ypred = predict(tree2,X_test);
        [SE(aa,bb),SP(aa,bb),ACC(aa,bb),BAC(aa,bb)] = compute_metrics(ypred, Y_test);
    end
    
end
figure();

subplot(2, 2, 1);
imagesc(SE);
colorbar;
xlabel("alpha");
ylabel("k-folds");
title('Sensitivity');

subplot(2, 2, 2);
imagesc(SP);
colorbar;
xlabel("alpha");
ylabel("k-folds");
title('Specificity');

subplot(2, 2, 3);
imagesc(BAC);
colorbar;
xlabel("alpha");
ylabel("k-folds");
title('Balanced Accuracy');

subplot(2, 2, 4);
imagesc(ACC);
colorbar;
xlabel("alpha");
ylabel("k-folds");
title('Accuracy');

colormap("jet");

[val,pos] = max(mean(BAC));
tree_pruned = prune(tree, "Alpha", alpha_grid(pos));
view(tree_pruned,'Mode','graph')

%% Evaluación del arbol podado mediante K-Fold

ypred = predict(tree_pruned, X(pos_test, :));
[SE_prune, SP_prune, BAC_prune, ACC_prune] = compute_metrics(ypred, Y(pos_test));
fprintf('BAC del árbol de clasificación podado (nodos terminales=%d) = %4.2f \n\n',sum(~tree_pruned.IsBranchNode), BAC_prune);


%% Usando bagging para reducir la varianza (Bootstrap Aggregation!)
 
rng(4);
N = 100;
tree_bagged = TreeBagger(N, X(pos_train, :), Y(pos_train), "NumPredictorsToSample","all", "Method","classification");

% Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
tmp_ypred = predict(tree_bagged, X(pos_test,:));
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;

[SE_bagging, SP_bagging, BAC_bagging, ACC_bagging] = compute_metrics(ypred, Y(pos_test));
fprintf('BAC del árbol de clasificación de bagging (Numero de árboles=%d) = %4.2f \n\n',tree_bagged.NTrees, BAC_bagging);


%{
%% Bagging Reduciendo la cantidad de predictores

rng(4);
numPredictorsToSample = size(X,2)/3;
mdl_RF = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample",numPredictorsToSample, "Method","classification");

% Evaluamos rendimiento en test del árbol bagged 
ypred = predict(mdl_RF,X(pos_test,:));
%MSE = mean((cell2mat(ypred)-Y(pos_test)).^2);
%fprintf('RMSE (TEST) del RF = %4.2f \n\n',sqrt(MSE));

%% Importancia de diferentes predictores y Out Of Box Error
%%%%%%%%%%%%%%%%%% TODO: No entiendo que hace esto del todo aun %%%%%%%%%%%%%%%%%%%%%%%%%%
rng(4);
numPredictorsToSample = size(X,2)/3;
mdl_RF_OOB = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample",numPredictorsToSample, "Method","classification", "OOBPredictorImportance","on");
imp = mdl_RF_OOB.OOBPermutedPredictorDeltaError;

figure;
bar(imp);
ylabel('Importancia');
xlabel('Predictores');
h = gca;
h.XTickLabel = mdl_RF_OOB.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% OOB error

plot(sqrt(oobError(mdl_RF_OOB)))
xlabel('Número de árboles');
ylabel('OOB RMSE');

err = oobError(mdl_RF_OOB, "Mode", "Ensemble");
fprintf('RMSE OOB del RF = %4.2f \n\n',sqrt(err));
 
%}
