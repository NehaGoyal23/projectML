filename=Gear1
N = normalize(filename)
B = table2array(N)
algo='SNE' 
data_column=B(:,1:end-5);
class_column=B(:,end);
reduceddata=compute_mapping(data_column,algo,30);
reduced_data_class=[reduceddata,class_column];
csvwrite('Gear1SNE30.csv', reduced_data_class)