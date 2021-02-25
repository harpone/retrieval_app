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


- loading an NGTPY index 5M images/ 25M items takes quite a while, about 3 mins...
- also model takes a long time to load on K80 GPU, about 2 min - quite a bit faster on my GTX1080Ti
- maybe use https