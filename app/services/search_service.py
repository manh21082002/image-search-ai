import numpy as np
import faiss
import pickle
import os

class ImageSearchEngine:
    def __init__(self,
                 index_path: str = "models/index.faiss",
                 name_path: str = "models/image_names.pkl",
                 base_image_dir: str = "data/cropped_images"):
        """
        Khởi tạo search engine từ FAISS index và danh sách tên ảnh.
        """
        self.index = faiss.read_index(index_path)

        with open(name_path, 'rb') as f:
            self.image_names = pickle.load(f)

        self.base_image_dir = base_image_dir

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Tìm top_k ảnh gần nhất với query_vector
        """
        query_vector = query_vector.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            img_path = os.path.join(self.base_image_dir, self.image_names[idx])
            results.append(img_path)

        return results
