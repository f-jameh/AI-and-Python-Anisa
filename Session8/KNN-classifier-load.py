# In the name of God

from joblib import load

clf = load('/home/farhad/knn_model.pkl')
print('model loaded')

new_sample = [[5.5,2.5,4.0,1.3]]

prediction = clf.predict(new_sample)
print(f'new_sample: {new_sample}')
print(f'predicted class: {prediction[0]}')

