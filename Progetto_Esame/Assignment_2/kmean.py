from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('Progetto_Esame/Assignment_2/sift_descriptors.pkl', 'rb') as f:
    sift_list = pickle.load(f)

print(sift_list[0].shape)