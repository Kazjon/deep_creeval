import theano, scipy
import numpy as np
from PIL import Image
from pylearn2.utils import serial

model_path = '/Users/kazjon/Dropbox/Documents/Research/UNCC/ComputationalCreativity/CCWorkshop/pylearn2/pylearn2/scripts/tutorials/stacked_autoencoders/dae_l1.pkl'
model = serial.load(model_path)
I = model.get_input_space().make_theano_batch()
O = model.get_output_space().make_theano_batch()
Y = model.encode(I)
Z = model.decode(O)
encode_funct = theano.function( [I], Y )
decode_funct = theano.function( [O], Z )

img = Image.open('mnist7.png')
x = np.reshape(img,[28*28,1])
recon_img = decode_funct(encode_funct(x.T))
scipy.misc.imsave('mnist7_recon.png',recon_img.reshape([28,28]))