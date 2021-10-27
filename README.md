This repo contains my implementation of the OpenMax module ([ref](https://vast.uccs.edu/~abendale/papers/0348.pdf)).
I reused some code from [here](https://github.com/ashafaei/OD-test/blob/8252aace84e2ae1ab95067876985f62a1060aad6/methods/openmax.py#L37).

Differences from other implementations I have found out there:
1. This code can be easily plugged in your pipeline.
2. This repo is documented.
3. This implementation is optimized (vectorized where it was possible).

### What's it for?
* OpenMax is a method for Open Set Classification (you have __K__ known classes and a set of classes the model hasn't seen during training which you have to classify as "unknown").
* OpenMax is applied only during __inference__ and on top of a network which was trained with SoftMax, for example.
* Its principle is based on the [Extreme Value Theory](https://en.wikipedia.org/wiki/Extreme_value_theory).
* The main idea is to build a probability model which would estimate how probable is that the sample belongs to one of the known classes
based on the distance from the sample's embedding to the class centroids which are estimated from the train data.
Based on this probability model OpenMax recalibrates the test samples' logits, also adding __K+1-th__ score which accounts for the "unknown" class.

### How OpenMax works
#### Phase 1: Weibull Fitting on the train data
__Input__: embeddings of __correctly__ classified samples for each class.
1. Compute centroids for each class
2. For each sample, compute the distance between its embedding and the respective centroid
3. Take __k__ farthest samples for each class
4. For each class, fit a Weibull distribution on these __k__ largest distances

#### Phase 2: Logits recalibration for the test data
__Input__: embeddings and logits of the test samples, precomputed Weibull models and class centroids.
1. Compute the distance between each sample and __each__ class centroid
2. Take __alpha__ closest centroids for each sample
3. Based on these __alpha__ closest centroids, for each sample recalibrate its logits according to the respective Weibull models
4. Compute probabilities as a SoftMax over the recalibrated logits
5. Classify a sample as "unknown" either if the respective probability is the largest or all of the probabilities fall below a threshold 

For more details please see the [paper](https://vast.uccs.edu/~abendale/papers/0348.pdf).

#### How to select a threshold
Threshold value may be selected so that a certain percent (99%, for example) of the train set is classified as "known".

#### Note
I believe it is not stated clearly in the paper, but you can use embeddings from any other layer than from the penultimate one
to fit the Weibull models. For example, the layer before the penultimate layer is a good choice for this, as embeddings in
this layer are trained to be linearly separable and it sort of makes sense.

#### On the usage of LibMR
In my experiments I have found ot that the results of fitting with [LibMR](https://github.com/Vastlab/libMR) differ from
 the results obtained with [this](https://github.com/ashafaei/OD-test/blob/8252aace84e2ae1ab95067876985f62a1060aad6/methods/openmax.py#L37)
implementation. Based on my own judgement I decided to use the latter (cdf of LibMR fitting looked less plausible for me).

### Usage
See `example.ipynb` for an example of use. Replace the toy data with the outputs of your model and here you go.
