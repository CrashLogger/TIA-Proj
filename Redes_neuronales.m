function Redes_neuronales

load Xtrain.mat
load Ytrain.mat

X = zscore(Xtrain);
Y = Ytrain;
BAC_media = [];
parfor i=1:10
    rng(i);
    [trainInd,valInd,testInd] = dividerand(length(Y),0.7,0.15,0.15);
    
    rng(2);
    k = 5;
    th = 0.5;
    c = cvpartition(length(trainInd)+length(valInd),'KFold',k);
    
    posCV = [trainInd valInd];posCV = sort(posCV);
    X1 = X(posCV,:);
    Y1 = Y(posCV);
    
    CV_error=[];N_grid = 2:2:50;
    for aa = 1:k
        pos_train = find(c.training(aa));
        pos_test_CV = find(c.test(aa));
        pos_val_CV = pos_train(1:length(pos_test_CV));
        pos_train_CV = pos_train(length(pos_test_CV)+1:end);
               
        for bb=1:length(N_grid)
            rng(13);
            net = patternnet(N_grid(bb));
            net.layers{1}.transferFcn = 'logsig';
            net.trainParam.showWindow = 0;
            net.divideFcn='divideind';
            net.divideParam.trainInd= pos_train_CV;
            net.divideParam.valInd= pos_val_CV;
            net.divideParam.testInd= pos_test_CV;
            net.inputs{1}.processFcns{2}='mapstd';
            
            train_net = train(net,X1',Y1');
            
            ypred = train_net(X1(pos_test_CV,:)');
            yfit=ypred;
            yfit(ypred>th)=1;
            yfit(ypred<=th)=0;
                    
            % CV_error(aa,bb) = 100*(1-(sum(yfit==Y1(pos_test_CV)')/length(Y1(pos_test_CV))));
            [~,~,~,BAC] = compute_metrics(yfit',Y1(pos_test_CV));
            CV_error(aa,bb) = 100*(1-BAC);
            yfit = [];
        end
        
    end
    
    [val,pos] = min(mean(CV_error));
    % fprintf("\nNumero de neuronas en capa intermedia con menor error:\n\tNeuronas = %d\n\tError = %4.2f%%\n",N_grid(pos),val);
    
    errorbar(N_grid,mean(CV_error),std(CV_error));hold on;plot(N_grid,mean(CV_error),'ro');hold off;xlabel('#neuronas');ylabel('Error CV');
    
    rng(13);
    
    net = patternnet(N_grid(pos));
    net.layers{1}.transferFcn = 'logsig';
    net.divideFcn='divideind';
    net.divideParam.trainInd= trainInd;
    net.divideParam.valInd= valInd;
    net.divideParam.testInd= testInd;
    net.inputs{1}.processFcns{2}='mapstd';
    
    train_net = train(net,X',Y');
    
    ypred = train_net(X(testInd,:)');
    yfit=ypred;
    yfit(ypred>th)=1;
    yfit(ypred<=th)=0;
    
    [~,~,~,BAC_media(i)] = compute_metrics(yfit',Y(testInd));
    fprintf("\nBAC%d: %f",i,BAC_media(i));
    % fprintf("\nSE: %4.2f\nSP: %4.2f\nACC: %4.2f\nBAC: %4.2f\n",SE,SP,ACC,BAC);
end

fprintf("\nBAC mediado: %f\n",sum(BAC_media)/10);