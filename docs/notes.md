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
- render output as HTML, not matplotlib figure; maybe a flashing point at the item location
- use domain neohuman.one
- use https
- link to original url for each retrieved example
- get rid of global variables with maybe a Result class or session or whatever