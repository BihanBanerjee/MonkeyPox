import os
import pandas as pd
from sklearn.utils import shuffle

def generate_csv(path, name):
  print("csv being generated")
  data = []
  dirs = ["Original Images"]
  uniques = ["Monkey Pox", "Others"]
  for dir in dirs:
    for unique in uniques:
      directory = path + "/" + dir + "/" + unique
      for filename in os.listdir(directory):
        paths = directory + "/" + filename
        data.append([ filename , paths  , unique])
  df = pd.DataFrame(data, columns = ["filename" ,"path", "class"])
  df = shuffle(df)
  
  df.to_csv(name, index = False)#name is the required path to store the new csv file containing all the images.
  print("Generation Complete")
  return df
