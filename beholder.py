# normal python
import os
import numpy as np
from time import time

# visualizing
import matplotlib.pyplot as plt
from skimage.io import imread
from tqdm import tqdm

# local imports
from dataset import cocoXYFilenames, GloveTransformer, Word2VecTransformer
from utils import ModelIO, npy2jpg
import pipeline
from config import MODEL_FILES_DIR, COCO_DIR, W2V_PATH, COCO_IMG

from gensim.models import Word2Vec


class Beholder(object):
    def __init__(self, data='train2014', encoder='project1.coco80k.jointembedder',
                 glove="glove.6B.%sd.txt.gz", tot=60000):
        # Load Image-Caption Data Stream
        print 7*'*' + ' INITIALIZING ASSETS ' + 7*'*'
        
        self.img_path = os.path.join(COCO_DIR, data)
        
        X, Y, self.filenames = cocoXYFilenames(dataType=data)
        sources = ("X", "Y")
        sources_k = ("X_k", "Y_k")


        images, captions = [],[]

        stream = pipeline.DataETL.getFinalStream(X, Y, sources=sources,
                    sources_k=sources_k, batch_size=tot)

        im, ca, _0, _1 = stream.get_epoch_iterator().next()
        images.append(im)
        captions.append(ca)
            
        images = np.vstack(images)
        captions = np.vstack(captions)
        
        encoder_name = encoder
        
        self.jointembedder_Txt = ModelIO.load(os.path.join(MODEL_FILES_DIR, encoder_name + "_Txt"))
        
        jointembedder_Img = ModelIO.load(os.path.join(MODEL_FILES_DIR, encoder_name + "_Img"))
        self.image_embs = jointembedder_Img(images)
        
        glove_version = glove % 300
        
        mod = Word2Vec.load(W2V_PATH)
        self.gloveglove = Word2VecTransformer(mod, None, pipeline.vect)#GloveTransformer(glove_version, None, pipeline.vect)
        
        print  9*'*' + ' READY TO SEARCH ' + 9*'*'
        
    def to_text_embedder(self, inputText):
        words = inputText.split(' ')
        word_seq_vect = []
        for word in words:
            try:
                word_seq_vect.append(
                    self.gloveglove.vectors[self.gloveglove.lookup[word]]
                    )
            except:
                print 'term', word, 'is OOV'
        word_seq_vect = np.vstack(word_seq_vect)
        word_seq_vect = word_seq_vect[None]
        return word_seq_vect
    
    def search(self, query, top=9, viz=True):
        st = time()
        word_rep = self.to_text_embedder(query)
        phrase_rep = self.jointembedder_Txt(word_rep)
        idx = find_images_from_query(self.image_embs, phrase_rep, top_n=top)
        
        paths = []
        for i in idx:
            img_file = npy2jpg(self.filenames[i])
            imp = os.path.join(self.img_path, img_file)
            paths.append(imp)

	paths = [os.path.join(COCO_IMG,p.split('/')[-1]) for p in paths]
        
        if viz: plot_gallery(paths)
        print 'executed query in', time()-st
        return paths
    
    
def get_similarities(imgs, query):
    # l2 norm the image embedding
    for i in range(len(imgs)):
        imgs[i] /= np.linalg.norm(imgs[i])

    # l2 norm the text embedding
    for i in range(query.shape[0]):
        query[i] /= np.linalg.norm(query[i])

    # cosine simliarity (the dot product of l2 normalized vectors)
    return np.dot(query, imgs.T).flatten()

def find_images_from_query(imgs, query, top_n=9):

    if query.shape[0] != 1:
        print("Query should only be a single phrase. \
               Got %d phrases " % query.shape[0])
        return

    sims = get_similarities(imgs, query)
    found = np.argsort(sims)[-top_n:]
    
    return reversed(found)

def plot_gallery(images, n_row=3, n_col=3):
    if n_row*n_col != len(images):
        n_col = len(images)/n_col + 1
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(4 * n_col, 4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)
        img = imread(images[i])
        plt.imshow(img)
        plt.xticks(())
        plt.yticks(())

