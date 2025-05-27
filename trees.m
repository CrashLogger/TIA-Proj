function trees
%TREES para el proyecto

%% PROBAR:
% Hacer bagging y comprobar con OOB en lugar de las metricas de erik
% Como es clasificación, es mejor hacerlo así:
% - OOB hace su propio error
% - No hace falta separar training y test!

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

BAC_ENTEROS = [];
BAC_PODADOS_KFOLD = [];

% Pongo 4 para aprovechar los 4 núcleos y no perder tiempo después
rngVals = linspace(1, 4, 4);
for r = rngVals

    rng(r);

    % 50/50 de train y test
    hpartition = cvpartition(size(X, 1), "HoldOut", 0.5);
    pos_train = hpartition.training;
    pos_test = hpartition.test;

    tree = fitctree(X(pos_train, :), Y(pos_train));
    alpha_grid = tree.PruneAlpha;

    %% Evaluación del arbol inicial

    %view(tree, "Mode", "graph");
    ypred = predict(tree, X(pos_test,:));
    %MSE = mean((Y(pos_test)-ypred).^2)
    [~,~,~,BAC_orig] = compute_metrics(ypred, Y(pos_test));
    %fprintf('BAC del árbol de clasificación entero (nodos terminales=%d) = %4.6f \n\n',sum(~tree.IsBranchNode), BAC_orig);
    BAC_ENTEROS = [BAC_ENTEROS, BAC_orig];

    %% Cross-Validation
    %  Usamos K-Fold porque no hay motivo para no usarlo

    rng(r)
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

    [~,pos] = max(mean(BAC));
    tree_pruned = prune(tree, "Alpha", alpha_grid(pos));
    %view(tree_pruned,'Mode','graph')

    %% Evaluación del arbol podado mediante K-Fold

    ypred = predict(tree_pruned, X(pos_test, :));
    [~, ~, ~, BAC_prune] = compute_metrics(ypred, Y(pos_test));
    %fprintf('BAC del árbol de clasificación podado (nodos terminales=%d) = %4.6f \n\n',sum(~tree_pruned.IsBranchNode), BAC_prune);
    BAC_PODADOS_KFOLD = [BAC_PODADOS_KFOLD BAC_prune];
end

disp("======================================================================")
fprintf("Media BAC arboles enteros: %4.6f\n", mean(BAC_ENTEROS));
disp("======================================================================")
fprintf("Media BAC arboles podados: %4.6f\n", mean(BAC_PODADOS_KFOLD));
disp("======================================================================")

pause();
disp("======================================================================")

%% Hacemos cross validation de Número de Árboles y menos predictores a la vez

%{
% ------------------------------------------------------------------------------------------------
% NO LO HACEMOS TODAS LAS VECES, ES PERDER EL TIEMPO! LOS RESULTADOS ESTÁN EN treeResultsCV.csv
% ------------------------------------------------------------------------------------------------

parfor r = rngVals

    Xtrain_CV = X
    Ytrain_CV = Y

    for nTrees = 10:10:120
        for nPredict = 1:1:45
            mdl_treebagger = TreeBagger(nTrees, Xtrain_CV, Ytrain_CV, "NumPredictorsToSample",numPredictorsToSample, "Method","classification", "OOBPredictorImportance","on");
            ooberroriter = oobError(mdl_treebagger);
            fileID = fopen('treeResultsCV.csv','a+');
            fprintf(fileID, "%d,%d,%d,%4.6f\n", r, nTrees, nPredict, ooberroriter(end));
            fprintf("%d,%d,%d,%4.6f\n", r, nTrees, nPredict, ooberroriter(end));
            fclose(fileID);
        end
    end
end

% ------------------------------------------------------------------------------------------------
%}


%% Entrenamos arbol con la cantidad de arboles y predictores conseguida mediante CV
%  Para ahorrar tiempo, tenemos los valores hard-codeados, obtenidos de una pasada de CV anterior
%  Los datos de esa pasada están en "treeResultsCV.csv"
% Hacemos el top 5 de los conseguidos y comparamos

numTrees = [100, 100, 100, 60, 70];
numPredictors = [45, 44, 6, 39, 27];

top5results = []

for r = rngVals
    for idx = 1:1:5
        tree_bagged = TreeBagger(numTrees(idx), X(pos_train, :), Y(pos_train), "NumPredictorsToSample",numPredictors(idx), "Method","classification");

        % Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
        tmp_ypred = predict(tree_bagged, X(pos_test,:));
        ypred = zeros(size(tmp_ypred));
        ypred(cell2mat(tmp_ypred) == '1') = 1;
        [~, ~, ~, BAC_bagging] = compute_metrics(ypred, Y(pos_test));

        fprintf('%d,%d,%d,%4.6f\n',r, numTrees(idx), numPredictors(idx), BAC_bagging);
        top5results(r,idx) = BAC_bagging;
    end
end

top5results

[~, pos] = max(mean(top5results));

best_tree_bagged = TreeBagger(numTrees(pos), X(pos_train, :), Y(pos_train), "NumPredictorsToSample",numPredictors(pos), "Method","classification");

rng(2025)
% Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
tmp_ypred = predict(best_tree_bagged, X(pos_test,:));
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;
[~, ~, ~, BAC_bagging] = compute_metrics(ypred, Y(pos_test));

fprintf('%d,%d,%d,%4.6f\n',r, numTrees(pos), numPredictors(pos), BAC_bagging);
top5results(r,idx) = BAC_bagging;

%% FALTA:
% Random forests!