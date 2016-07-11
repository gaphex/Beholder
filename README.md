##Requirements
------------
- Numpy
- Pandas
- Matplotlib
- Theano
- Blocks
- Fuel
- sklearn
- IPython
- Foxhound
- pycocotools

##Dependancies
I recommend installing the depedencies from the sources provided in /lib/ folder.

##Pre-trained Models
--------------------
###GloVe Vector Files
The path to the GloVe vectors (http://www-nlp.stanford.edu/data/glove.6B.300d.txt.gz) is configured in ```~/.fuelrc``` and should look something like ```~/datasets/glove/glove.6B.300d.txt.gz```.

###Image Features
Image features can be obtained using a CNN of choice (e.g. VGG*)

##Training Data
---------------
The location for MSCOCO data is configured through the ```config.py``` file. e.g. ```COCO_DIR=~/datasets/coco```.

Precomputed image features for MSCOCO should be stored at ```$COCO_DIR/features/train2014```, or ```$COCO_DIR/features/val2014```.  The images should be named exactly as the image files, with ```.jpg``` extension subsituted with ```.npy```.

Caption files should be located in ```$COCO_DIR/annotations```. For example, captions for the train2014 data should be located at ```$COCO_DIR/annotations/captions_train2014.json```.