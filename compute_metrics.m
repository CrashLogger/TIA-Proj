function [SE,SP,ACC,BAC] = compute_metrics(Ypred,Y)

%Compute SE, SP and BER
SE=sum(Ypred & Y)/sum(Y);
SP=sum(~Ypred & ~Y)/sum(~Y);
BAC = (SE+SP)/2;
ACC=(sum(Ypred & Y)+sum(~Ypred & ~Y))/(sum(Y)+sum(~Y));