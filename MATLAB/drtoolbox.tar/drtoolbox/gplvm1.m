filename=Gear1
B = table2array(filename)
algo='GPLVM' 
data_column=B(:,1:end-1);
class_column=B(:,end);
reduceddata=compute_mapping(data_column,algo,25)
reduced_data_class=[reduceddata,class_column];
csvwrite('Gear18GPLVM20.csv', reduced_data_class)