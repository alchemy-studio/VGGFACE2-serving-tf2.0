# VGGFACE2-serving-tf2.0
This project implements the serving code of VGGFACE2 facial recognition

## download model
vggface2 model is too large to upload to github, so its model file is save at google drive. download the model [here](https://drive.google.com/open?id=1N6GngCyvXL3fK2i4jbzGVTL2LSlXoFB8) and place the file under directory models.

## prepare facial database
add pictures of new identity by creating directory with the name of the identity under directory facedb. make sure every added picture contains only one person or the picture will be discarded when the KNN is trained.

## run demo
test the facial recognizer by executing
```bash
python3 OfflineRecognizer.py <path/to/video>
```

