function  [Z,pval]=trend_test(label,score)
%Input: label:N by 1 vectors of integers. Each row corresponds to an individual. Each entry
%             indicates the individual is in the i-th group (i=1,2,...,k).
%             The order of the integers corresponds to the increasing order of trend
%             being tested.
%       score: N by 1 matrix of summary score for tested.
%Output: Z: zscore of Jonckheere¨CTerpstra trend test statistic.
%        pval:p-value of Z.
% Programmed by Yunyi Zhou.
grp=unique(label);
ngrp=length(grp);
sel=nchoosek(1:ngrp,2);
J=0;
for i=1:size(sel,1)
    sel_g=sel(i,:);
    vec1=score(label==sel_g(1));
    vec2=score(label==sel_g(2));
    U=sum(sum((repmat(vec1,1,length(vec2))>repmat(vec2',length(vec1),1))));
    J=J+U;
end
N=length(label);
nrow=zeros(ngrp,1);
for g=1:ngrp
    nrow(g)=sum(label==grp(g));
end
Z=(J-(N^2-nrow'*nrow)/4)/sqrt((N^2*(2*N+3)-sum((nrow.^2).*(2*nrow+3)))/72);
pval=1-cdf('norm',Z,0,1);
end


