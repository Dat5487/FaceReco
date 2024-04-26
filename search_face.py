import pickle
from sklearn.metrics.pairwise import cosine_similarity


class SearchFace:
    def __init__(self):
        with open("data/embbedings.pkl", 'rb') as f:
            self.embeddings = pickle.load(f)
        with open(b"data/ids.pkl", "rb") as f:
            self.ids = pickle.load(f)

        self.threshold = 0.6


    def save_file(self,emb,id):
        
        self.embeddings.append(emb)
        self.ids.append(id)
        
        filehandler = open(b"data/embbedings.pkl", "wb")
        pickle.dump(self.embeddings, filehandler)
        filehandler.close()

        filehandler = open(b"data/ids.pkl", "wb")
        pickle.dump(self.ids, filehandler)
        filehandler.close()


    def search(self,emb):
        threshold_max = 0
        id_max = -1
        for idx, embedd in enumerate(self.embeddings):
            score = cosine_similarity(emb, embedd)
            print(score)
            if score > self.threshold:
                
                if score>threshold_max:
                    id_max = self.ids[idx]
                    threshold_max = score
        return id_max
# if __name__ == "__main__":
#     embeddings=[]
#     ids = []
#     filehandler = open(b"data/embbedings.pkl", "wb")
#     pickle.dump(embeddings, filehandler)
#     filehandler.close()

#     filehandler = open(b"data/ids.pkl", "wb")
#     pickle.dump(ids, filehandler)
#     filehandler.close()
