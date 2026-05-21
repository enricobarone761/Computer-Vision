import pickle
from sklearn.cluster import KMeans

with open(r'Progetto_Esame/Assignment_2/descrittori&vacabolario/descrittori_300.pkl', 'rb') as f:
    descriptors = pickle.load(f)

K_VALUES = [50, 100, 500]

for K in K_VALUES:
    print(f"Training KMeans con K={K}...")
    kmeans = KMeans(n_clusters=K, random_state=42, verbose=1)
    kmeans.fit(descriptors)
    print(f"  {K}-means addestrato con inertia: {kmeans.inertia_:.2f}")

    with open(rf'Progetto_Esame/Assignment_2/vocab_k{K}_300.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"  Vocabolario K={K} salvato in vocab_k{K}_300.pkl")