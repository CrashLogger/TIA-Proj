function trees
%TREES para el proyecto

load("Xtrain.mat")
load("Ytrain.mat")

close all;

% Se que es cualitativo y estoy usando MSE para los Cross-Validation.
% No se que me estaba pasando en la cabeza. Lo arreglaré pronto :tm:.

% Pese a que el archivo se llame [X/Y]train, voy a tratarlo como si fuera [X/Y] entero.

X = zscore(Xtrain);
Y = Ytrain;

% Después de ver esto:
% https://media.discordapp.net/attachments/942077236524253215/1375869434438222014/image.png?ex=683341c3&is=6831f043&hm=9368f96cc8008f044d15d8776f060f9a7b1afed2ce9575906af792b450e1c691&=&format=webp&quality=lossless&width=1250&height=704
% Podemos quitar sin ningún miramiento múltiples predictores

size(X,2)

useablePredictors = ones(size(X,2),1);
%useablePredictors(9) = 0;
useablePredictors(18) = 0;
%useablePredictors(25) = 0;
%useablePredictors(26) = 0;
%useablePredictors(35) = 0;
useablePredictors(37) = 0;
%useablePredictors(43) = 0;
useablePredictors(44) = 0;

logicaluseablePredictors = logical(useablePredictors)

X = X(:,logicaluseablePredictors);


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
fprintf('BAC del árbol de clasificación entero (nodos terminales=%d) = %4.6f \n\n',sum(~tree.IsBranchNode), BAC_orig);

%% Cross-Validation
%  Usamos K-Fold porque no hay motivo para no usarlo

rng(1)
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
fprintf('BAC del árbol de clasificación podado (nodos terminales=%d) = %4.6f \n\n',sum(~tree_pruned.IsBranchNode), BAC_prune);

%% Usando bagging para reducir la varianza (Bootstrap Aggregation!)
 
rng(1);

% Partimos los datos de training en 50/50
hpartition = cvpartition(size(X, 1), "HoldOut", 0.5);
pos_train = hpartition.training;
pos_test = hpartition.test;

X1 = X(pos_train,:);
Y1 = Y(pos_train);

N = 100;
tree_bagged = TreeBagger(N, X(pos_train, :), Y(pos_train), "NumPredictorsToSample","all", "Method","classification");

% Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
tmp_ypred = predict(tree_bagged, X(pos_test,:));
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;

[SE_bagging, SP_bagging, BAC_bagging, ACC_bagging] = compute_metrics(ypred, Y(pos_test));
fprintf('BAC del árbol de clasificación de bagging (Numero de árboles=%d) = %4.6f \n\n',tree_bagged.NTrees, BAC_bagging);

%% Haciendo CV con bagging para ajustar el número de arboles

stepSize = 10;
PossibleNTreess = 10 : stepSize : 200;
BAC_CV_bagging = zeros(size(PossibleNTreess));

hpartition = cvpartition(size(X1, 1), "HoldOut", 0.5);
pos_train_CV = hpartition.training;
pos_test_CV = hpartition.test;

for nTrees = PossibleNTreess
     
    tree_bagged = TreeBagger(nTrees, X1(pos_train_CV,:), Y1(pos_train_CV), "NumPredictorsToSample","all", "Method","classification");

    % Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
    tmp_ypred = predict(tree_bagged, X1(pos_test_CV,:));
    ypred = zeros(size(tmp_ypred));
    ypred(cell2mat(tmp_ypred) == '1') = 1;

    [~, ~, BAC_CV_bagging(nTrees/stepSize), ~] = compute_metrics(ypred, Y1(pos_test_CV));
    %%% DEBUG %%% fprintf('BAC del árbol de clasificación de bagging (Numero de árboles=%d) = %4.6f \n\n',tree_bagged.NTrees, BAC_CV_bagging(nTrees/stepSize));
end
figure()
stem((BAC_CV_bagging-mean(BAC_CV_bagging))/std(BAC_CV_bagging))
xlabel("x10      Número de Árboles")
title("Resultados de CV: BAC normalizada")

% Entrenamos arbol con la cantidad de arboles conseguida mediante CV

[val,pos] = max(BAC_CV_bagging);
tree_bagged = TreeBagger(PossibleNTreess(pos), X(pos_train, :), Y(pos_train), "NumPredictorsToSample","all", "Method","classification");

% Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
tmp_ypred = predict(tree_bagged, X(pos_test,:));
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;

[SE_bagging, SP_bagging, BAC_bagging, ACC_bagging] = compute_metrics(ypred, Y(pos_test));
fprintf('BAC del árbol de clasificación de bagging (Numero de árboles=%d) = %4.6f \n\n',tree_bagged.NTrees, BAC_bagging);


%% Bagging, peror educiendo la cantidad de predictores
rng(1);
size(X,2)
numPredictorsToSample = size(X,2)/3;
mdl_RF = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample",numPredictorsToSample, "Method","classification", "OOBPredictorImportance","on");

% Evaluamos rendimiento en test del árbol bagged 
tmp_ypred = predict(tree_bagged, X(pos_test,:));
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;

imp = mdl_RF.OOBPermutedPredictorDeltaError;

figure;
bar(imp);
ylabel('Importancia');
xlabel('Predictores');
h = gca;

[SE_bagging, SP_bagging, BAC_bagging, ACC_bagging] = compute_metrics(ypred, Y(pos_test));
fprintf('BAC del árbol de clasificación de bagging (Numero de árboles=%d, Numero de predictores=%d) = %4.6f \n\n',mdl_RF.NTrees, numPredictorsToSample, BAC_bagging);

%% Hacemos cross validation de Número de Árboles y menos predictores a la vez

stepSize = 10;
PossibleNTreess = 10 : stepSize : 200;
BAC_CV_bagging = zeros(size(PossibleNTreess));

numPredictorsToSample = linspace(1,size(X,2),size(X,2))

hpartition = cvpartition(size(X1, 1), "HoldOut", 0.5);
pos_train_CV = hpartition.training;
pos_test_CV = hpartition.test;

for nTrees = PossibleNTreess

    for nPredictors = numPredictorsToSample
        tree_bagged = TreeBagger(nTrees, X1(pos_train_CV,:), Y1(pos_train_CV), "NumPredictorsToSample",nPredictors, "Method","classification");

        % Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
        tmp_ypred = predict(tree_bagged, X1(pos_test_CV,:));
        ypred = zeros(size(tmp_ypred));
        ypred(cell2mat(tmp_ypred) == '1') = 1;

        [~, ~, BAC_CV_bagging(nTrees/stepSize, nPredictors), ~] = compute_metrics(ypred, Y1(pos_test_CV));
        %%% DEBUG %%% fprintf('BAC del árbol de clasificación de bagging (Numero de árboles=%d, Número de predictores=%d) = %4.6f \n\n',tree_bagged.NTrees, nPredictors, BAC_CV_bagging(nTrees/stepSize, nPredictors));
    end
end
figure()
imagesc(BAC_CV_bagging)
colormap("jet")
colorbar()
ylabel("x10      Número de Árboles")
xlabel("Número de Predictores")
% Entrenamos arbol con la cantidad de arboles y predictores conseguida mediante CV

max_val = max(BAC_CV_bagging(:))

linear_index = find(BAC_CV_bagging == max_val, 1);
[row, col] = ind2sub(size(BAC_CV_bagging), linear_index)

tree_bagged = TreeBagger(row*stepSize, X(pos_train, :), Y(pos_train), "NumPredictorsToSample",col, "Method","classification");

% Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
tmp_ypred = predict(tree_bagged, X(pos_test,:));
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;

row*stepSize
[SE_bagging, SP_bagging, BAC_bagging, ACC_bagging] = compute_metrics(ypred, Y(pos_test));
fprintf('BAC del árbol de clasificación de bagging (Numero de árboles=%d, Número de predictores=%d) = %4.6f \n\n',tree_bagged.NTrees, col, BAC_bagging);;

%% FALTA:
% Random forests!