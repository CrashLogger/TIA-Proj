function [SE,SP,ACC,BAC] = compute_metrics(Ypred,Y)

%Compute SE, SP and BER
SE=sum(Ypred & Y)/sum(Y);                                   % SENSITIVITY - Busca los positivos correctos - Mayor es mejor
SP=sum(~Ypred & ~Y)/sum(~Y);                                % SPECIFICITY - Busca los negativos correctos - Mayor es mejor
BAC = (SE+SP)/2;                                            % BALANCED ACCURACY - La media de las anteriores, util para datos no equilibrados (como los nuestros)
ACC=(sum(Ypred & Y)+sum(~Ypred & ~Y))/(sum(Y)+sum(~Y));     % ACCURACY - Los verdaderos positivos y los verdaderos negativos entre todos los resultados