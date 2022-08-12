## loading the dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv("iris.data",
                 names=['sepal_length','sepal_width','petal_length','petal_width','target'])
print(df)
print(pd.isna(df))
#print(df['sepal_length'])
missing_values_status = pd.isna(df)
number_of_missing_value_rows_for_sepal_length = 0
number_of_missing_value_rows_for_sepal_width = 0
number_of_missing_value_rows_for_petal_length = 0
number_of_missing_value_rows_for_petal_width = 0
number_of_missing_value_rows_for_target = 0
for i in missing_values_status['sepal_length']:
    if i:
        number_of_missing_value_rows_for_sepal_length += 1
for i in missing_values_status['sepal_width']:
    if i:
        number_of_missing_value_rows_for_sepal_width += 1
for i in missing_values_status['petal_length']:
    if i:
        number_of_missing_value_rows_for_petal_length += 1
for i in missing_values_status['petal_width']:
    if i:
        number_of_missing_value_rows_for_petal_width += 1
for i in missing_values_status['target']:
    if i:
        number_of_missing_value_rows_for_target += 1

print(number_of_missing_value_rows_for_sepal_length)
print(number_of_missing_value_rows_for_sepal_width)
print(number_of_missing_value_rows_for_petal_length)
print(number_of_missing_value_rows_for_petal_width)
print(number_of_missing_value_rows_for_target)
remove_missing_data = df.dropna()
print(remove_missing_data)

le = preprocessing.LabelEncoder()
le.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
#print(df['target'].values)
target_numerical_labels = le.transform(remove_missing_data['target'].values)
data = remove_missing_data
remove_missing_data_add_numerical_labels_for_target = data.replace(remove_missing_data['target'].values, target_numerical_labels)
print(remove_missing_data_add_numerical_labels_for_target)

data_description = remove_missing_data_add_numerical_labels_for_target.describe()

# mean before normalization
print(data_description['sepal_length'][1])
print(data_description['sepal_width'][1])
print(data_description['petal_length'][1])
print(data_description['petal_width'][1])
# std before normalization
print(data_description['sepal_length'][2])
print(data_description['sepal_width'][2])
print(data_description['petal_length'][2])
print(data_description['petal_width'][2])
target = list(remove_missing_data_add_numerical_labels_for_target['target'])
remove_missing_data_add_numerical_labels_for_target.drop('target', inplace=True, axis=1)

scaler = StandardScaler()
scaler.fit(remove_missing_data_add_numerical_labels_for_target)
remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists = scaler.\
    transform(remove_missing_data_add_numerical_labels_for_target)
print(remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists)
remove_missing_data_add_numerical_labels_for_target_normalize = remove_missing_data_add_numerical_labels_for_target.\
    replace(remove_missing_data_add_numerical_labels_for_target['sepal_length'].values,
            remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 0])
remove_missing_data_add_numerical_labels_for_target_normalize = \
    remove_missing_data_add_numerical_labels_for_target_normalize. \
    replace(remove_missing_data_add_numerical_labels_for_target['sepal_width'].values,
            remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 1])
remove_missing_data_add_numerical_labels_for_target_normalize = \
    remove_missing_data_add_numerical_labels_for_target_normalize. \
        replace(remove_missing_data_add_numerical_labels_for_target['petal_length'].values,
                remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 2])
remove_missing_data_add_numerical_labels_for_target_normalize = \
    remove_missing_data_add_numerical_labels_for_target_normalize. \
        replace(remove_missing_data_add_numerical_labels_for_target['petal_width'].values,
                remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 3])

print(remove_missing_data_add_numerical_labels_for_target_normalize)
data_description = remove_missing_data_add_numerical_labels_for_target_normalize.describe()

# mean after normalization
print(data_description['sepal_length'][1])
print(data_description['sepal_width'][1])
print(data_description['petal_length'][1])
print(data_description['petal_width'][1])
# std after normalization
print(data_description['sepal_length'][2])
print(data_description['sepal_width'][2])
print(data_description['petal_length'][2])
print(data_description['petal_width'][2])
features = PCA(n_components=2).fit_transform(remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, :4])
print(features)

colors = np.array(["blue", "red", "green"])
plt.scatter(features[:, 0], features[:, 1], c=colors[target])
plt.xlabel('dimension one')
plt.ylabel('dimension two')
plt.show()
plt.boxplot(remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 0])
plt.show()
plt.boxplot(remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 1])
plt.show()
plt.boxplot(remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 2])
plt.show()
plt.boxplot(remove_missing_data_add_numerical_labels_for_target_normalize_list_of_lists[:, 3])
plt.show()







