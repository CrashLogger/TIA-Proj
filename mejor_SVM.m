load Xtrain.mat;
load Ytrain.mat;

%% Usando 60% train / %40 test
BAC_media_SVC = [];
Xtrain = zscore(Xtrain);


%Quitar tres predictores de baja importancia segun lo visto en trees
useablePredictors = ones(size(Xtrain,2),1);
useablePredictors(18) = 0;
useablePredictors(37) = 0;
useablePredictors(44) = 0;

logicaluseablePredictors = logical(useablePredictors);

%Normalizar las entradas
Xtrain = Xtrain(:,logicaluseablePredictors);
for i=1:16
    
    rng(i);
    hpartition = cvpartition(length(Ytrain),'Holdout',0.40);
    pos_train = hpartition.training;
    pos_test = hpartition.test;
    
    x1 = Xtrain(pos_train,:);
    x2 = Xtrain(pos_test,:);
    y1 = Ytrain(pos_train);
    y2 = Ytrain(pos_test);
    
    %Pesos: Hay mas observaciones de una clase que de otra
    weights = ones(size(y1));
    weights(y1 == 0) = (sum(y1 == 1)/sum(y1 == 0));  % Peso clase 0
    weights(y1 == 1) = 1;  % Peso clase 1

    % Modelo con mejor C
    SVMModel = fitcsvm(x1, y1, "BoxConstraint", 0.8, ...
                      "KernelFunction", "linear",'weights',weights,"Cost",cost);
    
    % Evaluar
    label = predict(SVMModel, x2);
    [~,~,~,BAC_media_SVC(i)] = compute_metrics(label,y2);
    fprintf('RNG = %d Precisión del SVC = %.4f \n',i, BAC_media_SVC(i));


    
    
    % % Confusion matrix
    % figure();
    % C = confusionmat(y2, label);
    % confusionchart(C, {'Clase (0)', 'Clase (1)'})
    % pause; close;
    
    
end    
fprintf("\nBAC mediado SVC: %f\n",mean(BAC_media_SVC));