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
% https://media.discordapp.net/attachments/942077236524253215/1375869434438222014/image.png?ex=68368d83&is=68353c03&hm=37a8f9611116e84c10e841d0c97ebc3f5840b3e8d0c5de45c1830fe15a48b659&=&format=webp&quality=lossless&width=1250&height=704
% Podemos quitar sin ningún miramiento múltiples predictores

size(X,2);

useablePredictors = ones(size(X,2),1);

%{
 useablePredictors(1) = 0;
useablePredictors(8) = 0;
useablePredictors(9) = 0;
useablePredictors(18) = 0;
useablePredictors(25) = 0;
useablePredictors(26) = 0;
useablePredictors(35) = 0;
useablePredictors(37) = 0;
useablePredictors(43) = 0;
useablePredictors(44) = 0; 
%}


useablePredictors(2) = 0;
useablePredictors(10) = 0;
useablePredictors(30) = 0;

useablePredictors(45) = 0; 
useablePredictors(46) = 0;
useablePredictors(47) = 0;

logicaluseablePredictors = logical(useablePredictors);

X = X(:,logicaluseablePredictors);

fprintf("MATRIZ DE CORRELACIÓN:\n");
correlmx = corrcoef(X)

figure();
imagesc(correlmx);
colormap("jet")
colorbar()

numTrees = [100, 100, 100, 60, 70];
numPredictors = [38, 31, 6, 35, 27];

%{
 
[XClean, TF] = rmoutliers(X, "gesd");
size(XClean)
YClean = Y(~TF, :);


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

    view(tree, "Mode", "graph");
    ypred = predict(tree, X(pos_test,:));
    %MSE = mean((Y(pos_test)-ypred).^2)
    [~,~,~,BAC_orig] = compute_metrics(ypred, Y(pos_test));
    BAC_ENTEROS = [BAC_ENTEROS, BAC_orig];

    %% Cross-Validation
    %  Usamos K-Fold porque no hay motivo para no usarlo

    rng(r)
    k = 10;
    c = cvpartition(sum(pos_train),'KFold',k);

    X1 = X(pos_train,:);
    Y1 = Y(pos_train);

    BAC = [];

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
            [~,~,~,BAC(aa,bb)] = compute_metrics(ypred, Y_test);
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

% Random forest, vaya!

% ------------------------------------------------------------------------------------------------
% NO LO HACEMOS TODAS LAS VECES, ES PERDER EL TIEMPO! LOS RESULTADOS ESTÁN EN treeResultsCV.csv
% ------------------------------------------------------------------------------------------------

%{
parfor r = rngVals

    Xtrain_CV = X
    Ytrain_CV = Y

    for nTrees = 10:10:110
        for nPredict = 1:1:45
            mdl_treebagger = TreeBagger(nTrees, Xtrain_CV, Ytrain_CV, "NumPredictorsToSample",nPredict, "Method","classification", "OOBPredictorImportance","on");
            ooberroriter = oobError(mdl_treebagger);
            fileID = fopen('treeResultsCV-NO-OUTLIERS.csv','a+');
            fprintf(fileID, "%d,%d,%d,%4.6f\n", r, nTrees, nPredict, ooberroriter(end));
            fprintf("%d,%d,%d,%4.6f\n", r, nTrees, nPredict, ooberroriter(end));
            fclose(fileID);
        end
    end
end 
%}

% ------------------------------------------------------------------------------------------------

%% Entrenamos arbol con la cantidad de arboles y predictores conseguida mediante CV
%  Para ahorrar tiempo, tenemos los valores hard-codeados, obtenidos de una pasada de CV anterior
%  Los datos de esa pasada están en "treeResultsCV.csv"
% Hacemos el top 5 de los conseguidos y comparamos



top5results = [];

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

[~, pos] = max(mean(top5results));
pos = 2;

rng(1)
best_tree_bagged = TreeBagger(numTrees(pos), X(pos_train, :), Y(pos_train), "NumPredictorsToSample",numPredictors(pos), "Method","classification");


% Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
tmp_ypred = predict(best_tree_bagged, X(pos_test,:));
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;
[~, ~, ~, BAC_bagging] = compute_metrics(ypred, Y(pos_test));

C = confusionmat(Y(pos_test), ypred)
SE = C(1,1) / (C(1,1) + C(2,1));
SP = C(2,2) / (C(2,2) + C(1,2));
BAC = (SE + SP)/2;

figure();
confusionchart(C, {'Down (0)','Up (1)'});

fprintf('%d,%d,%d,%4.6f\n',r, numTrees(pos), numPredictors(pos), BAC_bagging);
%top5results(r,idx) = BAC_bagging;

disp("CV-ING THE RESULT WITH MULTIPLE HOLD OUTS")

given_BAC = [];

%% CV just for a sanity check
for ho = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    rng(1)
    c = cvpartition(size(XClean,1),'holdout',ho);
    pos_train_C = c.training;
    pos_test_C = c.test;

    rng(1)
    given_tree_bagged = TreeBagger(numTrees(pos), XClean(pos_train_C, :), YClean(pos_train_C), "NumPredictorsToSample",numPredictors(pos), "Method","classification");

    % Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
    tmp_ypred = predict(given_tree_bagged, X(pos_test, :));
    ypred = zeros(size(tmp_ypred));
    ypred(cell2mat(tmp_ypred) == '1') = 1;
   % [~, ~, ~, BAC_bagging] = compute_metrics(ypred, Y(pos_test));

    C = confusionmat(Y(pos_test), ypred);
    SE = C(1,1) / (C(1,1) + C(2,1));
    SP = C(2,2) / (C(2,2) + C(1,2));
    BAC = (SE + SP)/2;
    given_BAC = [given_BAC BAC];

    %figure();
    %confusionchart(C, {'Down (0)','Up (1)'});


    
end

disp(mean(given_BAC))

% %% Boosting

fprintf("BOOSTING");


rng(1)
ho = 0.5
c = cvpartition(size(XClean,1),'holdout',ho);
pos_train_C = c.training;
pos_test_C = c.test;

treeTemplate = templateTree();
boosted_tree = fitcensemble(XClean(pos_train_C, :), YClean(pos_train_C), 'Method','LogitBoost', 'Learners', treeTemplate, 'NumLearningCycles', 100);

tmp_pred = predict(boosted_tree, X);
ypred = zeros(size(tmp_ypred));
ypred(cell2mat(tmp_ypred) == '1') = 1;

C = confusionmat(Y(pos_test), ypred);
SE = C(1,1) / (C(1,1) + C(2,1));
SP = C(2,2) / (C(2,2) + C(1,2));
BAC = (SE + SP)/2;

fprintf('%2.2f,%4.6f\n',ho, BAC); 
%}

fprintf("# ===================================================================== #\n")
fprintf("                        ------ FINAL ------\n")
fprintf("# ===================================================================== #\n")

rng(1)

for pos_C = [3]%linspace(1, 5, 5)
    parfor r = linspace(1,64,64)
        rng(r)
        ho = 0.4;
        c = cvpartition(size(X,1),'holdout',ho);
        pos_train_C = c.training;
        pos_test_C = c.test;

        given_tree_bagged = TreeBagger(numTrees(pos_C), X(pos_train_C, predictorCount), Y(pos_train_C), "NumPredictorsToSample",numPredictors(pos_C), "Method","classification");

        % Binarizamos a mano porque por algún motivo tree_bagged devuelve un cell array donde cada cell tiene un caracter
        tmp_ypred = predict(given_tree_bagged, X(pos_test_C, predictorCount));
        ypred = zeros(size(tmp_ypred));
        ypred(cell2mat(tmp_ypred) == '1') = 1;
        % [~, ~, ~, BAC_bagging] = compute_metrics(ypred, Y(pos_test));

        C = confusionmat(Y(pos_test_C), ypred);
        SE = C(1,1) / (C(1,1) + C(2,1));
        SP = C(2,2) / (C(2,2) + C(1,2));
        BAC = (SE + SP)/2;

        fileID = fopen('final2.csv','a+');
        fprintf('%d,%d,%d,%4.6f\n', r, numTrees(pos_C), predictorCount, BAC);
        fprintf(fileID, '%d,%d,%d,%4.6f\n', r, numTrees(pos_C), predictorCount, BAC);
        fclose(fileID);

        %figure();
        %confusionchart(C, {'Down (0)','Up (1)'});
    end
end

%%%%%%
%%MODELO PARA DEMO
%%%%%%

finalTree = TreeBagger(100, X, Y, "NumPredictorsToSample",6, "Method","classification");