import pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.base import clone

def train_kmeans_model(nfeatures):
    with open(rf'Progetto_Esame/Assignment_2/descrittori_e_vacabolario/descrittori_{nfeatures}.pkl', 'rb') as f:
        descriptors = pickle.load(f)

    K_VALUES = [50, 100, 500]
    kmeans = KMeans(random_state=42, verbose=1)

    for K in K_VALUES:
        print(f"Training KMeans con K={K}...")
        km = clone(kmeans)
        km.set_params(n_clusters=K)
        km.fit(descriptors)

        with open(rf'Progetto_Esame/Assignment_2/descrittori_e_vacabolario/vocab_k{K}_{nfeatures}.pkl', 'wb') as f:
            pickle.dump(km, f)
        print(f"Vocabolario K={K} salvato")


if __name__ == "__main__":
    train_kmeans_model(1000)