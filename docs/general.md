# Reverse Image Search

There has been a lot of hype about "AI" recently. While we may not be getting fully self driving cars just yet, not to mention AGI robots, one thing that we're going to see a lot more in the future is a "[Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35)" trend. One feature of this is that complex, high dimensional data will be able to be compressed into a low dimensional representation, that will contain maximal amount of semantically relevant information. For example, it is possible to represent the contents of a high dimensional image as a short "code" learned by a deep convolutional neural network.



## Idea in a nutshell:

Given a user image or a specific item/entity in an image, the service will retrieve similar images or images with same/similar items/entities from a database.

Similar to Google Lens, but we would offer an API or a SaaS, which could return a "code" for each image URL/path/etc.

Aimed e.g. towards web stores of any size. Nowadays big web stores roll their own search applications, but smaller ones don't really have very good options.

I.e. given `https://adogslifephoto/labrador-retriever-europees-hond.jpg`, the app will return a 

`code=010100011101010011111` 

*or* a JSON containing

```
    {'item_0': 
        {'category': 'animal',
         'class': 'dog, labrador',
        'segmentation': [...],
         'code': '01010010101001010010'}}}
     'item_1':
         {'category': 'furniture',
         'class': 'couch',
        'segmentation': [...],
         'code': '11101000001011010011'}}
```

The binary codes will allow 1) efficient storage and 2) fast retrieval by e.g. Multi-Index Hashing.

We could provide an efficient API or even a command line program, which anyone could use to build a database and implement the search themselves.

The codes are extracted by a Deep Learning model that has been specifically trained to extract such codes, which contain maximal information about the image/items in image, and can therefore be used in similarity search. See e.g here:

[Arponen, Bishop - "Learning to hash with semantic similarity metrics and empirical KL divergence"](https://arxiv.org/abs/2005.04917)

## Business model

- Lincensing the code
- Building a SaaS: 
    - Level 1: feed in image urls, get codes and customer can use those however they please
    - Level 2: feed in image urls => we handle the database, user queries and retrieval results - customer will simply display them on their website/webstore
- Releasing an Android and iOS app
- Releasing an executable for all major operating systems


## Similar products/ companies

- Google Vision Product Search: https://cloud.google.com/vision/product-search/docs
    - This is pretty obscure and Google will get your data (check if they *have* to do this)
- Google Lens: only offered as an app
- ViSenze: https://www.visenze.com/
    - Strong at least in research


## Tech approach

### DL research and development:
Start with a good pretrained, fully connected model => saves a lot of research & dev time and cost. Codes can be extracted directly from a layer below the actual prediction => get a model almost immediately for testing/demos.

**Need to emphasize that representations will be the software 2.0!**

### SaaS/ web app:
This could be started immediately in parallel to the R&D effort


# Results

## Encoding benchmarks:

### Home:
Specs: 4x Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz, 32GB RAM DDR3
2/4xCPU: openimages_50k_0.h5: images=105, codes=342 :: 636 seconds
- both processes at 200%!
4xCPU: openimages_50k_1.h5: images=100, codes=359 :: 1197 seconds
- all 4 processes running at about 100% only, 12% RAM

1 CPU (400%): 35s
2 CPU (200%): 40s
4 CPU (100%): 39s


### neo-1:
Specs: n1-standard-4 (4 vCPUs, 15 GB memory)
- 3 processes running (4 submitted)
- processes at 120-140% CPU, 25% RAM (RAM bound?)

4xvCPU: openimages_50k_3.h5: images=97, codes=341 :: 1007 seconds
16xvCPU: openimages_50k_3.h5: images=7, codes=28 :: 677 seconds
4/16xvCPU: openimages_50k_0.h5: images=15, codes=39 :: 453 seconds
- 4 processes at 400% each; 5% RAM

4: 0.291 img/s
16: 0.165 img/s
4/16: 0.13 img/s

### neo-2:
Specs: e2-standard-4 (4 vCPUs, 16 GB memory)
- 3 processes running (4 submitted)
- processes at 120-140% CPU, 25% RAM (RAM bound?)

4xvCPU: openimages_50k_0.h5: images=111, codes=361 :: 1144 seconds


## More:
2vCPU, 1 process: 
Recorded: 09:01:45  Samples:  46378
Duration: 1508.973  CPU time: 2912.328
   
16vCPU, 1 process:
- CPU at about 400%-1000%
- very close to 8x faster!
Recorded: 09:45:26  Samples:  44769
Duration: 182.110   CPU time: 774.095

4CPU (home), 1 process:
Recorded: 11:54:22  Samples:  32814
Duration: 168.339   CPU time: 573.617

1080Ti, same dataloading process:
Recorded: 21:52:24  Samples:  18274
Duration: 46.805    CPU time: 120.070
- note about 10s is model load overhead
- 10-20s image_from_url
=> can do maybe 3 img/s on my GPU 
=> about 24 img/s on 8xK80 maybe
=> about 2M img/day on 8xK80
=> abou $3.2 to $3.6/h for 8xK80 => about $75..$100

1080Ti, 0 workers: images=59, codes=191 :: 45 seconds
1080Ti, 1 workers: images=59, codes=191 :: 20 seconds
1080Ti, 2 workers: images=59, codes=191 :: 20 seconds

## Some stats:
- 27397 images found out of 30000, which is ~91%
- 92k codes, i.e. ~3.4 codes/image
- => 1000 images + codes ~ 1MB on disk; 1M ~ 1GB; 1B ~ 1TB
- can reduce size a lot by hashing
- lot of the codes are non-things; then most are person, car, dog, cat
- new version doing ~2 img/s
- 150k takes about 24h


# Competition

## Syte AI
https://www.syte.ai

- We do self supervised learning; they can do that too of course, but probably heavily invested/ spent tons of money & time in supervised learning + labeling solutions (they started in 2015 and most of their employees seem to be "data specialists")
- We'll be full remote => faster growth
- They are Israeli; we'll be EU first (?)
- Their industry report: https://drive.google.com/file/d/1cj9kSi3sxuxrzcdcVpLiZ5w9tEYNKKgi/view?usp=sharing
- they seem to offer tailor made solutions to web shops - we could offer a library or an API for a search engine
- Ofer Fryman: "Co-founder and CEO of Syte Visual Conception with 22 years of passion in Machine learning and deep learning" - almost entirely accounting experience, no publications => liar
- Science guy Dr. Helge Voss has lots of experience in trad. data science/ML in relation to particle physics and apparently about 5 years in DL so seems decent
- Maor Nissan - seems pretty good
- [Alex Lagre](https://www.linkedin.com/in/axel-lagr%C3%A9-41322043/) seems to be the DL guy (took part in Open Images challenge!)
- List of patents (NOT theirs): https://www.syte.ai/patents/

#### More benchmarks:
home: images=1000, codes=4804 :: 446 seconds
colab V100: images=1000, codes=4800 :: 576 seconds

=> colab is 1.7 img/s => 150k img/24h

# Company names:
Neohuman
Metapixel
Daimon


# Deployment:
**NOTE** `bel-2` is now the server (configured to start nginx at boot)

Following this: https://pythonise.com/series/learning-flask/deploy-a-flask-app-nginx-uwsgi-virtual-machine

### Preliminaries:
- In the Firewall section, tick both Allow HTTP traffic and Allow HTTPS traffic DONE
- if uwsgi install problem: first `conda config --add channels conda-forge`, then `conda install uwsgi`
- werkzeug has recent but: `pip install -U Werkzeug==0.16.0`
- uwsgi doesn't seem to work with webcam locally, so just launch the app directly on localhost
- on VM: `sudo ufw allow 9090`, `sudo ufw enable`
- WARNING `sudo ufw enable` gives "Command may disrupt existing ssh connections. Proceed with operation (y|n)?" That's what might be interfering with my SSH?? Maybe `sudo ufw disable` after done? Or `sudo ufw delete allow 9090`
- not OK to have ./env/bin inside app folder: https://serverfault.com/questions/957084/failed-at-step-exec-spawning-permission-denied so need to replace `Environment="PATH=/home/<yourusername>/app/env/bin"
ExecStart=/home/<yourusername>/app/env/bin/uwsgi --ini app.ini`



# Cloud solutions:
- vast.ai: pretty cheap! Could use the notebook interface...
- need github token with pull only prems
- need service account json with storage read/write only
- dedicated 1xK80 VM is about $0.5/h => $12/day


# OpenImages:
- OK around 200-10k images per "interesting class" (clothes, products) so should probably be enough for n-way smax classification, but try out few shot segmentation later


## Google Cloud Vision API/ product search:
https://cloud.google.com/vision/product-search/docs/quickstart

### Detect objects:
- requires GCP project & auth
- depends heavily on Google Knowledge Graph and labels
- image contents need to be sent as base64 encoded strings to the API (REST, python, ...)
- oh, can work on urls as well BUT the request URL needs to be inside a JSON with other args
- black box

### Product search:
- need a product catalog - API creates product set and does indexing, supports queries
- all in GCP naturally
- black box
- "Currently Vision API Product Search supports the following product categories: homegoods, apparel, toys, packaged goods, and general"

### Our offering:
- We could offer a library/ docker container with actual model etc... no cloud dependency
- and/or a drop-in replacement
- transparent box!
- use own model?

# TODO:
- Superhero?
- Tempo rahoitus?

# vast.ai
- restricted github token?
- restricted service account?

# TODOs:
- maybe use compression (blosc) with pytables... let's see the file sizes first

# Encoding notes:
- GTX 1080 is def fast enough, no need for lots of memory
- ATM total about 380k images in GCS (from sets 3 and 4)
- On a 6xGTX1080 with 1.25 img/s per GPU => 37h for 1M images = 28$




