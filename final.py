import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
with open('rf_tuned.pkl','rb') as f:
    model=pickle.load(f)
    input = np.array([20,1,28,0,1,3]).reshape(1,6)
    p=model.predict(input)
    print(p[0])
 