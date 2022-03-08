# The Microsoft Research Sentence Completion Challenge

```
The Microsoft Research Sentence Completion Challenge (Zweig and Burges, 2011) requires a system to 
be able to predict which is the most likely word (from a set of 5 possibilities) to complete a sentence. In
the labs you have evaluated using unigram and bigram models. In this assignment you are expected to
investigate at least 2 extensions or alternative approaches to making predictions. Your solution does
not need to be novel. You might choose to investigate 2 of the following approaches or 1 of the following
approaches and 1 of your own devising.

•Tri-gram (or even quadrigram) models
•Word similarity methods e.g., using Googlenews vectors or WordNet?
•Combining n-gram methods with word similarity methods e.g., distributional smoothing?
•Using a neural language model?
```

In this repository, I intend to create a jupyter notebook (likely using google colab) where I will implement various solutions to the challenge.

## Random

A random baseline is always a good idea, as this will show us wether our solutions are even worth trying.

## N-grams

My first solution will be with n-grams. I will create a dynamic n-gram constructor that is able to create n-grams of varying sizes, which will allow me to find an optimal depth to search back in training data, and I will then test this model on a testing set to determine its accuracy.

## Word-Net Similarity

WordNet provides 3 different similarity measures. It may be an innacurate solution, but comparing the similarity of nouns in the set with the possible words may have some level of success. This will be easy to perform, as we can find the average similarity between all the noun combinations with a very small amount of code.

## Neural Language Models

A neural language model may provide more accurate results, but this will require the most experimentation as internal values will need to be optimised for the problem.
