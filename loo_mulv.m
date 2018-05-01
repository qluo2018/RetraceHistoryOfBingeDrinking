function [acc,score,sel_fc,sel_snp,sel_para]=loo_mulv(label,fc,snp,cov)
%Input: label:N by 1 binary (0-1) vector indicating the group of each
%             individual,0 control, 1 dinker.
%       fc: N by p1 matrix containing functional connectivity of each individual
%       snp: N by p2 matrix containing SNP of each individual.
%       cov: N by p3 matrix containing covariates of each individual (N: sample
%       size.)
%
%Output: acc: a scalar indicating classification accurarcy by
%        leave-one-out.
%        score: a N by 4 matrix containing summary scores. Columns 1 to 4 corresponds to increased FC, decreased FC, risk
%        SNP, protective SNP.
%        sel_para: a vector containing final decided parameters.
% Programmed by Yunyi Zhou.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Grids for selecting parameters are pre-defined as follows. Users can
%change according to their needs.
C=0.005:0.005:0.1; %Boxconstant for SVM.
lamda=0.00005:0.00001:0.0002; %Lambda for lasso regression
topn=100:100:400;% Number of most significant FCs
pthr=4e-4:2e-5:7e-4;%Threshold for p-values of SNPs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inpcom=[fc,snp,cov];
N=length(label);
p2=size(snp,2);
ac=zeros(length(C),length(topn),length(lamda),length(pthr));
recfetfc=cell(length(C),length(topn),length(lamda),length(pthr));
rcfcsub=cell(N,1);
recfetsnp=cell(length(C),length(topn),length(lamda),length(pthr));
rcsnpsub=cell(N,1);
p1=size(fc,2);
preout=zeros(N,1);
nc=length(C);
ftop=length(topn);
nlmd=length(lamda);
npt=length(pthr);
indx=1:N;
psnp_sto=zeros(N,p2);
opt=statset('MaxIter',2*50000);
for i=1:N
    psnp=psnp_sto(i,:);
    leflab=label;
    leflab(i)=[];
    indxout=indx;
    indxout(i)=[];
    snpin=snp(indxout,:);
    parfor i_snp=1:p2
        [~,~,psnp(i_snp)]=crosstab(leflab,snpin(:,i_snp));
    end
    psnp_sto(i,:)=psnp;
end
%parpool(20)
for c=1:nc
    for tp=1:ftop
        for f=1:nlmd
            for pt=1:npt
                parfor i=1:N
                    lefout=inpcom(i,:);
                    indxout=indx;
                    indxout(i)=[];
                    lefin=inpcom(indxout,:);
                    leflab=label;
                    leflab(i)=[];
                    trfcctrl=lefin(leflab==0,1:p1);
                    trfc14=lefin(leflab==1,1:p1);
                    [~,p14]=ttest2(trfcctrl,trfc14);
                    [~,rp]=sort(p14);
                    fet14=rp(1:topn(tp));
                    B14=lassoglm([trfcctrl(:,fet14);trfc14(:,fet14)],leflab,'binomial','Lambda',lamda(f));
                    fet14=fet14(B14~=0);
                    rcfcsub{i}=fet14;
                    trfcctrl=trfcctrl(:,fet14);
                    trfc14=trfc14(:,fet14);
                    mctrl=mean(trfcctrl);
                    m14=mean(trfc14);
                    lgfc=find(mctrl>m14);
                    smfc=find(mctrl<m14);
                    lgfcctrl=sum(trfcctrl(:,lgfc),2);
                    lgfc14=sum(trfc14(:,lgfc),2);
                    smfcctrl=sum(trfcctrl(:,smfc),2);
                    smfc14=sum(trfc14(:,smfc),2);
                    selsnp=find(psnp_sto(i,:)<pthr(pt));
                    smctrl=mean(lefin(leflab==0,p1+selsnp));
                    sm14=mean(lefin(leflab==1,p1+selsnp));
                    lg=find(smctrl>sm14);
                    sm=find(smctrl<sm14);
                    lgctrl=sum(lefin(leflab==0,p1+selsnp(lg)),2);
                    lg14=sum(lefin(leflab==1,p1+selsnp(lg)),2);
                    smctrl=sum(lefin(leflab==0,p1+selsnp(sm)),2);
                    sm14=sum(lefin(leflab==1,p1+selsnp(sm)),2);
                    rcsnpsub{i}=selsnp;
                    svm14=svmtrain(cat(1,[lefin(leflab==0,[(p1+p2+1):end]),lgfcctrl,smfcctrl,lgctrl,smctrl],...
                        [lefin(leflab==1,[(p1+p2+1):end]),lgfc14,smfc14,lg14,sm14]),...
                        leflab,'kernel_function','linear','boxconstraint',C(c),'options',opt,'autoscale',true);
                    preout(i)=svmclassify(svm14,[lefout([(p1+p2+1):end]),sum(lefout(fet14(lgfc))),sum(lefout(fet14(smfc))),...
                        sum(lefout(p1+selsnp(lg))),sum(lefout(p1+selsnp(sm)))]);
                end
                ac(c,tp,f,pt)=sum(preout==label)/N;
                recfetfc{c,tp,f,pt}=rcfcsub;
                recfetsnp{c,tp,f,pt}=rcsnpsub;
            end
        end
    end
end
actop=max(max(max(max(ac))));
acc=actop;
selfet=find(ac==actop);
selfet=floor(median(selfet));
[i1,i2,i3,i4]=ind2sub([nc,ftop,nlmd,npt],selfet);
sel_para=[C(i1),topn(i2),lamda(i3),pthr(i4)];
fetfc=recfetfc{selfet};
fetsnp=recfetsnp{selfet};
fetfc=cell2mat(fetfc');
fetsnp=cell2mat(fetsnp');
fetfcfreq=tabulate(fetfc);
fetsnpfreq=tabulate(fetsnp);
% topfc=fetfcfreq(fetfcfreq(:,2)>=ceil(N*0.9),1:2);
% topsnp=fetsnpfreq(fetsnpfreq(:,2)>=ceil(N*0.9),1:2);
sel_fc=fetfcfreq(fetfcfreq(:,2)>=ceil(N*0.9),1);
sel_snp=fetsnpfreq(fetsnpfreq(:,2)>=ceil(N*0.9),1);
fcctrl=fc(label==0,sel_fc);
fcdrker=fc(label==1,sel_fc);
mctrl=mean(fcctrl);
mdrker=mean(fcdrker);
lgfc=find(mctrl>mdrker);
smfc=find(mctrl<mdrker);
iFC=sum(fc(:,smfc),2);
dFC=sum(fc(:,lgfc),2);
snpctrl=snp(label==0,sel_snp);
snpdrker=snp(label==1,sel_snp);
mctrl=mean(snpctrl);
mdrker=mean(snpdrker);
lgsnp=find(mctrl>mdrker);
smsnp=find(mctrl<mdrker);
rSNP=sum(snp(:,smsnp),2);
pSNP=sum(snp(:,lgsnp),2);
score=[iFC,dFC,rSNP,pSNP];
end
