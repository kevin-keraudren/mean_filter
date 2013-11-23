#!/usr/bin/python

import irtk
from lib.mean_filter import mean_filter

img = irtk.imread( "data/fetus_head.nii.gz", dtype='float32' )

irtk.imshow( img, filename="before.png" )

img = irtk.Image( mean_filter(img, 5, 5, 5 ),
                  header=img.get_header() )

irtk.imshow( img, filename="after.png" )

