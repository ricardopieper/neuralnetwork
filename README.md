# neuralnetwork

### Obsolete: This project became ProphetService (https://github.com/ricardopieper/ProphetService), a background service for a web app.

Neural Network using FLENS and OpenBLAS. 
Inspired on the Coursera's Machine Learning material (Andew Ng's Course).

This neural network tries to recognize handwritten digits, which is the "hello world" for neural networks.
It doesn't have any fancy UI to write the digits, it just runs the prediction algorithms.

I also ported the fmincg function from Octave. It seems to be working OK so far.

The neural network itself has 3 layers (input, hidden, output). 

The back-propagation algorithm is fully vectorized. However, it has to load a lot of things into memory. The algorithm is likely
to change in the future to address this problem.
