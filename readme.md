# Visual Entity Search app

This is a (more or less abandoned) visual search app to be run on a GCP VM. The user submits a query image, selects the "entity" (person, animal, clothes, item...) to be searched for from the database and receives top most similar entities from images in the database. It uses a self-supervised deep learning model ([SimCLR](https://github.com/google-research/simclr)) as the backbone. 

Supervised DL models suffer from "overfitting to the domain", which results in poor generalization outside of the training data domain. Using a supervised net for image retrieval etc. task would typically require continuous labeling of new training data. The reason for using a self-supervised model here is to make the app generalize better to unseen/new data, i.e. you can skip the continuous collection of new labelled data!

I'm also using [detectron2](https://github.com/facebookresearch/detectron2) to split the image into different "entities" (but not using the classes). I planned to do this step in a completely unsupervised fashion, but damn detectron works too well for this! :sweat_smile:

I'm using [NGTPY](https://github.com/yahoojapan/NGT/blob/master/python/README-ngtpy.md) to do approximate nearest neighbor search by using the SimCLR representations (projected to 128 dimensions with PCA), and [pytables](https://github.com/PyTables/PyTables) for a simple, fast vector database. Nowadays I'd probably use Milvus or Qdrant for the search part.

FYI I haven't included here how to set up the VM, uwsgi/nginx, database etc so you have to figure that out if you really want to try this :/

## How it works:
Upload an image:

![](https://i.imgur.com/BzXKR0G.png)

The model will split the image into "entities" (you could filter out the sky, leaves etc but keeping those here for now).

Then if you click on "1" (the dog "Hugo"), you get similar dog results:
![](https://i.imgur.com/YfL4otl.jpg)

The bullseye shows the location of the best match. Yeah I know, terrible UX! :joy: 

