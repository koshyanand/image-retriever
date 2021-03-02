"""

 similar_images_TL.py  (author: Anson Wong / git: ankonzoid)

 We find similar images in a database by using transfer learning
 via a pre-trained VGG image classifier. We plot the 5 most similar
 images for each image in the database, and plot the tSNE for all
 our image feature vectors.

"""
import sys, os
import numpy as np
import pickle
from keras.preprocessing import image
from keras.models import Model
sys.path.append("src")

from vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from imagenet_utils import preprocess_input
from plot_utils import plot_query_answer
from sort_utils import find_topk_unique
from kNN import kNN
from tSNE import plot_tsne
from keras import backend as K


def main():
    # ================================================
    # Load pre-trained model and remove higher level layers
    # ================================================
    print("Loading RESNET-50 pre-trained model...")    
    base_model = ResNet50(weights= 'imagenet', include_top=False)
    model = Model(input=base_model.input,
                  output=base_model.output)

    # ================================================
    # Read images and convert them to feature vectors
    # ================================================
    imgs, filename_heads, X = [], [], []
    path = "testData"
    print("Reading images from '{}' directory...\n".format(path))
    
    file = open('Matchlist_testData.txt', 'w')
    
    itr = 1 
    imgNameList = os.listdir(path)
    for f in imgNameList:
        # Process filename
        filename = os.path.splitext(f)  # filename in directory
        filename_full = os.path.join(path,f)  # full path filename
        head, ext = filename[0], filename[1]
        if ext.lower() not in [".jpg", ".jpeg"]:
            continue

        # Read image file
        print("IMAGE: {}".format(head))
        img = image.load_img(filename_full, target_size=(224, 224))  # load
        imgs.append(np.array(img))  # image
        filename_heads.append(head)  # filename head

        # Pre-process for model input
        img = image.img_to_array(img)  # convert to array
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img).flatten()  # features
        X.append(features)  # append feature extractor

        # Change the filename to normal indicies
        dst = path + "/" + str(itr) + ".jpg"
        src = path + "/" + f
        print(src, " --> ", dst)
        os.rename(src, dst)
        itr += 1


    print("The images are given indices as follows:\n")
    for i in range(0, len(imgs)):
        print("{} --> {}\n".format(filename_heads[i], i + 1))
    
        

    X = np.array(X)  # feature vectors
    imgs = np.array(imgs)  # images
    print("imgs.shape = {}".format(imgs.shape))
    print("X_features.shape = {}\n".format(X.shape))

    # ===========================
    # Find k-nearest images to each image
    # ===========================
    n_neighbours = 5 + 1  # +1 as itself is most similar
    knn = kNN()  # kNN model
    knn.compile(n_neighbors=n_neighbours, algorithm="brute", metric="cosine")
    knn.fit(X)

    # ==================================================
    # Plot recommendations for each image in database
    # ==================================================
    output_rec_dir = os.path.join("testData", "output", "rec")
    if not os.path.exists(output_rec_dir):
        os.makedirs(output_rec_dir)
    n_imgs = len(imgs)
    ypixels, xpixels = imgs[0].shape[0], imgs[0].shape[1]
    for ind_query in range(n_imgs):

        # Find top-k closest image feature vectors to each vector
        print("[{}/{}] Plotting similar image recommendations for: {}".format(ind_query+1, n_imgs, filename_heads[ind_query]))
        distances, indices = knn.predict(np.array([X[ind_query]]))
        distances = distances.flatten()
        indices = indices.flatten()
        indices, distances = find_topk_unique(indices + 1, distances, n_neighbours)
        print(indices, "\n", distances)
        
        for idx in np.squeeze(indices)[1:]:
            file.write(str(idx) + " ")
        file.write("\n")

        # Plot recommendations
        rec_filename = os.path.join(output_rec_dir, "{}_rec.png".format(filename_heads[ind_query]))
        x_query_plot = imgs[ind_query].reshape((-1, ypixels, xpixels, 3))
        x_answer_plot = imgs[indices - 1].reshape((-1, ypixels, xpixels, 3))
        plot_query_answer(x_query=x_query_plot,
                          x_answer=x_answer_plot[1:],  # remove itself
                          filename=rec_filename)

    # ===========================
    # Plot tSNE
    # ===========================
    output_tsne_dir = os.path.join("testData", "output")
    if not os.path.exists(output_tsne_dir):
        os.makedirs(output_tsne_dir)
    tsne_filename = os.path.join(output_tsne_dir, "tsne.png")
    print("Plotting tSNE to {}...".format(tsne_filename))
    plot_tsne(imgs, X, tsne_filename)

# Driver
if __name__ == "__main__":
    main()

K.clear_session()