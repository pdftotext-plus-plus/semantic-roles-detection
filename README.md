# How to run the Keras model in C++ Tensorflow

(1) **Training**: Train your model by using `train.py`. At the end of the training, the model will
    be stored in an `*.h5` file (which is in HDF5 format).
(2) **Convert `*.h5` to .pb**: Convert the `*.h5` file produced in the previous step to a *.pb file
    by using the `keras_to_tensorflow.py` script. The usage is as follows:

    python3 keras_to_tensorflow.py --model IChXO-model.h5 --numout 14 --prefix IChXO --name IChXO-model.pb