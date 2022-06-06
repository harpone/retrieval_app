# TODO

+ upload img as bytes, don't save to disk because => conflicts
  + well not bytes, but PIL image object as global variable... a bit ugly I think
  + see here for b64encode way: 
    + https://www.pythonanywhere.com/forums/topic/5017/
    + https://github.com/AnandVetcha/Flask_App/blob/master/templates/output.html
+ support PNG etc input
+ nice HTML pages:
  + use nicer styles
  + drag & drop or click to upload photo on landing page
  + ditch webcam for now, but think how to implement
+ need dev db & index - index paired with db name
+ render output as HTML, not matplotlib figure; maybe a flashing point at the item location
+ link to original url for each retrieved example
+ use domain neohuman.one
+ get rid of global variables with maybe a Result class or session or whatever
+ placeholder missing image in get_retrieval_plot
+ if uploaded_image == None, raise error or flash message! (exceeds 10MB upload limit)
- fucking large image prob... not resolved!!
  - maybe use JS to resize on device, then upload? A bit tricky... leave for later
  + going with simple upload button for now
- now 3 cols on mobile portrait, 2 on landscape :/
- can't handle pics that are actually short videos...


+ also model takes a long time to load on K80 GPU, about 2 min - quite a bit faster on GTX1080Ti
+ on CPU: 4CPU/40G RAM: $0.269 hourly, about $200/month - works barely, a bit slow - shit still sometimes OOM
- maybe try modern CPUs
- drop background classes!
- maybe use https


## Improved model:
- Better self-sup pretrained backbone
- Test feature diffusion!
- Top coding and segmentation nets OR feature diffusion etc

## Optimize index
- loading an NGTPY index 5M images/ 25M items takes quite a while, about 3 mins...
- maybe implement hashing to reduce RAM
- Actually, if codes are uniform in [0, 1] then simple scale and cast to uint8 should already be good and 4x smaller than float32 (only!!)
  - not sure if e.g. NGTPY support uint8... but supports float16 I think