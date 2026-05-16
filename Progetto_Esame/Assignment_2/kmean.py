import pickle
from sklearn.cluster import KMeans

with open('C:\\Users\\enrib\\Documents\\Ing_Informatica_LM32\\Computer-Vision\\descrittori.pkl', 'rb') as f:
    descriptors = pickle.load(f)

K_VALUES = [50, 100, 500]

for K in K_VALUES:
    print(f"Training KMeans con K={K}...")
    kmeans = KMeans(n_clusters=K, random_state=42, verbose=1)
    kmeans.fit(descriptors)
    print(f"  {K}-means addestrato con inertia: {kmeans.inertia_:.2f}")
    
    with open(f'vocab_k{K}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"  Vocabolario K={K} salvato in vocab_k{K}.pkl")