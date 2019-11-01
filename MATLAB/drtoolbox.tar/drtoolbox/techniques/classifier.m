clear;
accuracy = zeros;
recall = zeros;
precision = zeros;
F_score = zeros;
NewCols = zeros;
for k=1:2
    file = sprintf('Gear%dSNE30.csv', k);
    data = xlsread(file);
    label = data(:, end);
    dataset = data(:, 1:end-1);
    NewCols = [];
    
    for i = 1:size(dataset,2)
        if (norm(dataset(:,i)) ~= 0)
            NewCols = [NewCols i];
        end
    end
    
    dataset = dataset(:,NewCols);
    X = normalize(dataset,1);
    X=X';
    K=8;
    d=12;
    [D,N] = size(X);
    
    [data_new, out] = fa(dataset, 30);
    data_new = normalize(data_new);
    [row, column ] = size(data_new);
    indices = crossvalind('Kfold',row,10);
    
    for i = 1:10
        test = (indices == i);
        train = ~test;
        mdl = fitctree(data_new(train,:), label(train,:),'MinLeafSize',100,'MinParentSize',40);
        prelabel = predict(mdl, data_new(test, :));
        accuracy(i,k) = (length(find(prelabel == label(test,:)))/ length(prelabel)*100)';
        confMat = confusionmat(prelabel, label(test,:));
        %% calculation of precision, recall and F-score
        %%% recall
        for m =1:size(confMat, 1);
            recall(m)=confMat(m,m)/sum(confMat(m,:));
        end
        recall(isnan(recall))=[];
        Recall=sum(recall)/size(confMat,1);
        
        %%% précision
        for n =1:size(confMat,1);
            precision(n)=confMat(n,n)/sum(confMat(:,n));
        end
        precision(isnan(precision))=[];
        Precision=sum(precision)/size(confMat,1);
        %%% F-score
        F_score(i,k)=(100*2*Recall*Precision/(Precision+Recall))';
    end
    file
end
file = 'accuracy.xlsx';
xlswrite(file, accuracy);
file = 'F_score.xlsx';
xlswrite(file, F_score);