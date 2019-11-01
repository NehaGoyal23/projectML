clear;
close all;
data_num=xlsread('Gear21.xlsx');
fea_raw=data_num(:,1:end-1);
Y=data_num(:,end);

std_raw=std(fea_raw);
fea_sec=fea_raw(:,std_raw>1e-4);

X=zscore(fea_sec);
num_samples=size(fea_raw,1);
part=cvpartition(Y,'Holdout',0.20);
X_train=X(part.training,:);
Y_train=Y(part.training);
X_val=X(part.test,:);
Y_val=Y(part.test);
algo='SNE'; 
mappedX_train=compute_mapping(X_train,algo,10);
reduced_data_class=[X_train,Y_train];
csvwrite('21GearSNE10.csv', reduced_data_class)