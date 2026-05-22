import pickle
from sklearn.cluster import KMeans

def train_kmeans_model(nfeatures):
    with open(rf'Progetto_Esame/Assignment_2/descrittori&vacabolario/descrittori_{nfeatures}.pkl', 'rb') as f:
        descriptors = pickle.load(f)

    K_VALUES = [50, 100, 500]

    for K in K_VALUES:
        print(f"Training KMeans con K={K}...")
        kmeans = KMeans(n_clusters=K, random_state=42, verbose=1)
        kmeans.fit(descriptors)

        with open(rf'Progetto_Esame/Assignment_2/vocab_k{K}_{nfeatures}.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
            print(f"Vocabolario K={K} salvato")


if __name__ == "__main__":
    train_kmeans_model(300)