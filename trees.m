function trees
%TREES para el proyecto

load("Xtrain.mat")
load("Ytrain.mat")


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
MSE = mean((Y(pos_test)-ypred).^2)
fprintf('RMSE del árbol de clasificación (nodos terminales=%d) = %4.2f \n\n',sum(~tree.IsBranchNode),sqrt(MSE));

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
        CV_MSE(aa,bb) = mean((ypred-Y_test).^2);
    end
    
end
[val,pos] = min(mean(CV_MSE));
tree_pruned = prune(tree, "Alpha", alpha_grid(pos));
view(tree_pruned,'Mode','graph')

%% Evaluación del arbol podado mediante K-Fold

CV_RMSE = sqrt(CV_MSE);
ypred = predict(tree_pruned, X(pos_test, :));
MSE = mean((ypred-Y(pos_test)).^2);

% Conseguimos un árbol mucho mas simple con las mismas capacidades de predicción que el arbol complejo original!
fprintf('RMSE (TEST) del árbol podado (alpha=%.3f  nodos terminales=%d) = %4.2f \n\n',alpha_grid(pos),sum(~tree_pruned.IsBranchNode),sqrt(MSE));

%% Usando bagging para reducir la varianza (Bootstrap Aggregation!)


rng(4);
mdl_bagged = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample","all", "Method","classification");

close all;

ypred = predict(mdl_bagged, X(pos_test,:));
MSE = mean((cell2mat(ypred)-Y(pos_test)).^2);
fprintf('RMSE (TEST) del árbol bagged = %4.2f \n\n',sqrt(MSE));

%% Bagging Reduciendo la cantidad de predictores

rng(4);
numPredictorsToSample = size(X,2)/3;
mdl_RF = TreeBagger(100, X(pos_train, :), Y(pos_train), "NumPredictorsToSample",numPredictorsToSample, "Method","classification");

% Evaluamos rendimiento en test del árbol bagged 
ypred = predict(mdl_RF,X(pos_test,:));
MSE = mean((cell2mat(ypred)-Y(pos_test)).^2);
fprintf('RMSE (TEST) del RF = %4.2f \n\n',sqrt(MSE));

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