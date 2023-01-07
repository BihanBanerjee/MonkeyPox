import numpy as np
import cv2
def encode_y(y):
  Y = []
  for i in y : 
    if(i == "Monkey Pox" ):
      Y.append(0)
    elif(i == "Others" ):
      Y.append(1)
  return  np.array(Y).astype("float32")          

# convert file paths info nums 
#then normalize

def process_x(x):
   # return np.array([cv2.imread(i) for i in x ]).astype("float32")
   a = np.array([cv2.imread(i) for i in x ]).astype("float32")
   return np.array([cv2.resize(i, (224, 224)) for i in a])