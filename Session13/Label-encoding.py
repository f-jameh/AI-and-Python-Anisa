# In the name of God

##################### Integer (simple) label encoding
from sklearn.preprocessing import LabelEncoder

# sample labels
labels1 = ['smal', 'medium', 'larg', 'xlarge']

# create an object from LabelEncoder
label_encoder = LabelEncoder()

# 1) fitting labels
fitted_labels = label_encoder.fit(labels1)

# 2) transform labels to integers
transformed_label = label_encoder.transform(labels1)

print(transformed_label)

# total) fit and transform in same step
final_trans_labels = label_encoder.fit_transform(labels1)
print(final_trans_labels)

#################### One-hot Encoding
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df_labels = pd.DataFrame(labels1)

# create an object for OneHot
encoder = OneHotEncoder(sparse_output=False)

# 1) fitting labels
fitted_onehot = encoder.fit(df_labels)

# 2) transform labels to integers
transformed_onehot = encoder.transform(df_labels)
print(transformed_onehot)

# total) fit and transform in same step
final_onehot = encoder.fit_transform(df_labels)
print(final_onehot)















