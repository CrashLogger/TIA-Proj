load Xtrain.mat;
load Ytrain.mat;

%% Usando 60% train / %40 test
rng(1);
hpartition = cvpartition(length(Ytrain),'Holdout',0.40);
pos_train = hpartition.training;
pos_test = hpartition.test;

%Normalizar las entradas
Xtrain = zscore(Xtrain);


x1 = Xtrain(pos_train,:);
x2 = Xtrain(pos_test,:);
y1 = Ytrain(pos_train);
y2 = Ytrain(pos_test);


rng(2)
k = 10;
c = cvpartition(length(y1),'KFold',k);

%% SVC lineal (BAC = 0.9438)



C_grid = linspace(0.1,5,50);

%Coste de fallar, mas costoso fallar la clase 0 (2.2 ~ observaciones clase 1 / observaciones clase 0)
cost = [0 2.2;1 0];

%Pese a que se sigue llamando CV_error se usa para guardar el BAC que es lo que se valora
CV_error=[];
for aa = 1:k
    
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain_CV = x1(pos_train_CV,:);
    Xtest_CV = x1(pos_test_CV,:);
    Ytrain_CV = y1(pos_train_CV);
    Ytest_CV = y1(pos_test_CV);

    for bb = 1:length(C_grid)
           
            SVMModel = fitcsvm(Xtrain_CV, Ytrain_CV, "BoxConstraint", C_grid(bb), ...
                      "KernelFunction", "linear","Cost",cost);
            label = predict(SVMModel, Xtest_CV);
            [~, ~, ~, BAC] = compute_metrics(label, Ytest_CV);  
            CV_error(aa,bb) = BAC;
 
    end
end

%Valores de BAC con diferentes C
figure()
plot(C_grid,mean(CV_error));

%Al ser BAC, necesitamos el MAXIMO
[val,pos] = max(mean(CV_error));



%Pesos: Hay mas observaciones de una clase que de otra
weights = ones(size(y1));
weights(y1 == 0) = sum(y1 == 1)/sum(y1 == 0);  % Peso clase 0
weights(y1 == 1) = 1;  % Peso clase 1

% Modelo con mejor C
SVMModel = fitcsvm(x1, y1, "BoxConstraint", C_grid(pos), ...
                  "KernelFunction", "linear","Weights",weights,"Cost",cost);

% Evaluar
label = predict(SVMModel, x2);
[~,~,~,BAC] = compute_metrics(label,y2);
fprintf('Precisión del SVC (C=%.3f) = %.4f \n', C_grid(pos), BAC);

% Confusion matrix
figure();
C = confusionmat(y2, label);
confusionchart(C, {'Clase (0)', 'Clase (1)'})
pause; close;





%% SVM gaussiano (BAC = 0.9234)


C_grid = [0.1 1 2 4 6 8 10];

KS_grid = [1 2 3 4 5 6]; 
CV_error = [];


% Parallelize the outer loop
for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain_CV = x1(pos_train_CV,:);
    Xtest_CV = x1(pos_test_CV,:);
    Ytrain_CV = y1(pos_train_CV);
    Ytest_CV = y1(pos_test_CV);
   
    for bb = 1:length(C_grid)
        for cc = 1:length(KS_grid)
            SVMModel = fitcsvm(Xtrain_CV, Ytrain_CV, "BoxConstraint", C_grid(bb), ...
                              "KernelFunction", "gaussian", ...
                              "KernelScale", KS_grid(cc),"Cost",cost);
            label = predict(SVMModel, Xtest_CV);
            [~,~,~,BAC] = compute_metrics(label,Ytest_CV);
            CV_error(bb,cc,aa) = BAC;
        end
    end
    
end

CV_medios = mean(CV_error, 3);
%Al ser BAC, necesitamos el MAXIMO
[val, pos] = max(CV_medios(:));
[row, col] = ind2sub(size(CV_medios), pos);

%Pesos: Hay mas observaciones de una clase que de otra
weights = ones(size(y1));
weights(y1 == 0) = sum(y1 == 1)/sum(y1 == 0);  
weights(y1 == 1) = 1;  

%Modelo con mejor C y KS
SVMModel = fitcsvm(x1, y1, 'BoxConstraint', C_grid(row), ...
                  'KernelFunction', 'gaussian', ...
                  'KernelScale', KS_grid(col),"Weight", weights,"Cost",cost);

% Evaluar
[label, scores] = predict(SVMModel, x2);
[SE,SP,ACC,BAC] = compute_metrics(label,y2);
fprintf('Precisión de la SVM (C=%.3f  KS=%.3f) = %.4f \n', ...
        C_grid(row), KS_grid(col), BAC);

% Confusion matrix
figure()
C = confusionmat(y2, label);
confusionchart(C, {'Clase (1)', 'Clase (2)'})
pause; close;


%% Polinomios (BAC = 0.9419) / El mejor es de grado 1==> Lineal (SVC)


C_grid = [0.25,0.5,0.75,1];

%Grado del polinomio
P_grid = [1,2,3,4,5,6]; 
CV_error=[];

for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain_CV = x1(pos_train_CV,:);
    Xtest_CV = x1(pos_test_CV,:);
    Ytrain_CV = y1(pos_train_CV);
    Ytest_CV = y1(pos_test_CV);
    
    for bb = 1:length(C_grid)
        for cc = 1:length(P_grid)
            SVMModel = fitcsvm(Xtrain_CV, Ytrain_CV, "BoxConstraint", C_grid(bb), ...
                              "KernelFunction", "polynomial", ...
                              "PolynomialOrder", P_grid(cc),"Cost",cost);
            label = predict(SVMModel, Xtest_CV);
            [~,~,~,BAC] = compute_metrics(label,Ytest_CV);
            CV_error(bb,cc,aa) = BAC;
            
        end
    end
end

CV_medios = mean(CV_error, 3);
%Al ser BAC, necesitamos el MAXIMO
[val, pos] = max(CV_medios(:));
[row, col] = ind2sub(size(CV_medios), pos);

plot(P_grid,CV_medios);xlabel('C');ylabel('Error');legend('C= 0.25','C=0.5','C=0.75','C=1');

%Pesos: Hay mas observaciones de una clase que de otra
weights = ones(size(y1));
weights(y1 == 0) = sum(y1 == 1)/sum(y1 == 0);  
weights(y1 == 1) = 1;  

%Modelo con mejor C y P
SVMModel = fitcsvm(x1, y1, 'BoxConstraint', C_grid(row), ...
                  'KernelFunction', 'polynomial', ...
                  'PolynomialOrder', P_grid(col),"Weight", weights,"Cost",cost);

% Evaluar
[label, scores] = predict(SVMModel, x2);
[SE,SP,ACC,BAC] = compute_metrics(label,y2);
fprintf('Precisión de la SVM (C=%.3f  P=%.3f) = %.4f \n', ...
        C_grid(row), P_grid(col), BAC);

% Confusion matrix
figure()
C = confusionmat(y2, label);
confusionchart(C, {'Clase (1)', 'Clase (2)'})
pause; close;

