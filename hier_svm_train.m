function [acc_final,output,svm_layer,svm_final]=hier_svm_train(label,feature,D)
%Input: label: N by 1 binary (0-1) vector of label of each individual.
%       feature: N by p matrix, each row corresponds to an individual and
%       each column corresponds to a feature.
%       D: p by 1 vector of integers indicating the feature domain of each column of
%       feature.
%Output: acc_final: a scalar indicating accuracy of the final model.
%        output: N by 1 vector indicates the predicted
%        label by the final model.
%        svm_layer:the svms in the first layer
%        svm_final:the svm in the second layer
% Programmed by Yunyi Zhou.
dom=unique(D);
Nd=length(dom);
C=[0.001,0.005,0.01,0.05,0.1,0.5,1:0.5:3];
gama=1:0.5:4;
lc=length(C);
lg=length(gama);
N=length(label);
svm_score=zeros(N,Nd);
opt=statset('MaxIter',2*50000);
svm_layer=cell(1,Nd);
for nd=1:Nd
    inp=feature(:,D==dom(nd));
    acc=zeros(lc,lg);
    for c=1:lc
        for g=1:lg
            pre=zeros(N,1);
            parfor n=1:N
                lab_in=label;
                lab_in(n)=[];
                inp_out=inp(n,:);
                inp_in=inp(setdiff(1:N,n),:);
                svm=svmtrain(inp_in,lab_in,'kernel_function','rbf','boxconstraint',...
                    C(c),'options',opt,'autoscale',true,'tolkkt',1e-5,'rbf_sigma',gama(g));
                pre(n)=svmclassify(svm,inp_out);
            end
            acc(c,g)=sum(pre==label)/N;
        end
    end
    sel_para=floor((find(acc<=median(reshape(acc,lg*lc,1)),1,'last')));
    [i1,i2]=ind2sub([lc,lg],sel_para);
    svm=svmtrain(inp,label,'kernel_function','rbf','boxconstraint',...
        C(i1),'options',opt,'autoscale',true,'tolkkt',1e-5,'rbf_sigma',gama(i2));
    svm_layer{nd}=svm;
    sv=svm.SupportVectors;
    alpha=svm.Alpha;
    icp=svm.Bias;
    shift=svm.ScaleData.shift;
    scale=svm.ScaleData.scaleFactor;
    kfun=svm.KernelFunction;
    kfunargs=svm.KernelFunctionArgs;
    parfor i=1:N
        svm_score(i,nd)=alpha'*feval(kfun,sv,(inp(i,:)+shift).*scale,kfunargs{:})+icp;
    end
end
acc=zeros(1,lc);
for c=1:lc
    pre=zeros(N,1);
    parfor n=1:N
        lab_in=label;
        lab_in(n)=[];
        inp_out=svm_score(n,:);
        inp_in=svm_score(setdiff(1:N,n),:);
        svm=svmtrain(inp_in,lab_in,'kernel_function','linear',...
            'boxconstraint',C(c),'autoscale',...
            true,'options',opt,'tolkkt',1e-5);
        pre(n)=svmclassify(svm,inp_out);
    end
    acc(c)=sum(pre==label)/N;
end
sel_para=floor((find(acc<=median(acc),1,'last')));
svm=svmtrain(svm_score,label,'kernel_function','linear','boxconstraint',...
    C(sel_para),'autoscale',true,'options',opt,'tolkkt',1e-5);
output=svmclassify(svm,svm_score);
acc_final=sum(output==label)/N;
svm_final=svm;
end
