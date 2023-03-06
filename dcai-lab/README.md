# Lab assignments for Introduction to Data-Centric AI

This repository contains my solutions to the lab assignments for the [Introduction to
Data-Centric AI](https://dcai.csail.mit.edu/) class.

## Lab 1: Data-Centric AI vs. Model-Centric AI

The [first lab assignment](https://github.com/hazrulakmal/deep-learning/tree/main/dcai-lab/1.%20data-centric-model-centric) walks you through an ML task of building a text classifier, and illustrates the power (and often simplicity) of data-centric approaches.

[Notes-Lab 1]()
[Lec-1]()

# Lab 2: Label Errors

[This lab assignment]() guides you through writing your own implementation of
automatic label error identification using Confident Learning, the technique taught in today’s lecture.


[lec-2](https://dcai.csail.mit.edu/lectures/label-errors/)

## Lab 3: Dataset Creation and Curation

[This lab assignment]() is to analyze an already collected dataset labeled by multiple annotators.

How do we measure the correctness of a label ? by knowing 
1. A **consensus label** for each example that aggregates the individual annotations.
2. A **quality score for each consensus label** which measures our confidence that this label is correct.
    - Aggrement rate for each sample 
3. A **quality score for each annotator** which estimates the overall correctness of their labels.
    - Assume that the consensus label is correct, measure the accuracy of the annotator (aggrement rate compared to the rest of the annotators)

Relying on consesus label alone may have its own drawbacks:
1. Resolving ties is ambiguous in majority vote.
2. A bad annotator and good annotator have equal impact on the estimates.

A technique like CROADLAB (Classifier Refinement Of croWDsourced LABels) is meant to take those two reasons into account when deciding the true label. CROWDLAB estimates are based on the intuition that we should rely on the classifier’s prediction more for examples that are labeled by few annotators but less for examples labeled by many annotators (with many annotations, simple agreement already provides a good confidence measure). If we have trained a good classifier, we should rely on it more heavily but less so if its predictions appear less trustworthy. 

Caveats:
1. Since CROWDLAB partly relies on the classifier to break the ties (annotator's quality is also considered), one may want to produce a good classifier first as the quality of the technique relies havily on how good you classifier is in the first place. A bad classifier may end up choosing the wrong labels. 

[Lec & Notes-3](https://dcai.csail.mit.edu/lectures/dataset-creation-curation/)

## Lab 4: Data-centric Evaluation of ML Models

[This lab assignment]() is to try improving the performance of a given
model solely by improving its training data via some of the various strategies covered here.

1. Identifying Influencial points(outliers aka OOD samples) that may degrade model performance
2. Identifying `dificult to predict` classes (usually the minority class, classes that are hard to decompose (likely similar))
3. Evaluation/Test samples are representative of the population and are correctly labelled (refer label erros & data curation). 
4. Not restricting to a single metrics (may not tell about the performance for different classes)

[Lec & Notes-4](https://dcai.csail.mit.edu/lectures/data-centric-evaluation/)

## Lab 5: Class Imbalance, Outliers, and Distribution Shift]

[The lab assignment]() for this lecture is to implement and compare different methods for identifying outliers. For this lab, we've focused on anomaly detection. You are given a clean training dataset consisting of many
pictures of dogs, and an evaluation dataset that contains outliers (non-dogs). Your task is to implement and compare various methods for detecting these outliers. You may implement some of the ideas presented in today's
lecture, or you can look up other outlier detection algorithms in the linked references or online.

Subpopulation performance vs Class Imbalance are two different things but usually interconnected.
- Subpopulation performance cares about the slice/subset of the dataset and may not be the label - e.g medical diagnosis - model diagnosis prediction is better for male than woman (so not necessary because of the class imbalance)

### Dealing with Class Imbalance (Some classes are more prevalance than the rest)
1. Choosing the right metrics (precision, recall, f-beta score) - choosing higher beta means more weights are given to recall and less weights to precision, beta = 1 (harmonic mean - equal weights to both precision and recall)
2. Undersampling & Oversampling (different data may yield different results)
3. Synthetic data - still have the sample labels but features may different slightly.
4. Sample weight on Loss
5. Balanced mini-batch sampling - instead of randomly sample mini-batch, we oversample/undersample some classes so that the proportion of classes are equal.

### Dealing with Ouliers/Anamoly (data points that dont fit with the rest of the points in the datset)
Casues -> bad sensors/gaps in data(missing values)/rare events. 
Ussually, it's hard diffentiate between rare events and bad data so usually we would find the outliers first then differente them wether they are bad data or not

**Tasks** 
1. Outlier detection - given the whole dataset, predict which samples are different than the others, given the whole dataset (animays + random, non-animal stuff), seperate them. 
2. Anomaly detection - we know in-distribution samples, given the dataset, seperate the ID and OOD samples. given ID (animaly), the whole dataset (animays + random, non-animal stuff), predict the animals

Techniques
1. Simple interquatile range. 
2. Z-score. looking at the extreme tails as outliers
3. Isolation Forest
4. KNN distance
5. Autoencoder representation loss (autoencoder - reduce the dimensionility representation into a smaller dimension latent representation and can recontruct it back into its high dimensional representation)

### Dealing with distributional, covariate and concept shift
Distributional shift -> P(x, y)_train != P(x, y)_test
Covariate shift -> P(x)_train != P(x)_test but P(y|x)_train == P(y|x)_test
Concept shift -> P(y|x)_train != P(y|x)_test but P(x)_train == P(x)_test 

1. Monitor data - anomoly detection, if input is anomaly then model prediction may not be reliable
2. Monitor model

Techniques
1. Collect more data so that anomaly become more prevalance
2. Retrain model

[Lec & Notes-5](https://dcai.csail.mit.edu/lectures/imbalance-outliers-shift/) 

## Lab 6: Growing or Compressing Datasets

Focus on what to label?

[This lab][lab-6] guides you through an implementation of active learning.

[lab-6]: growing_datasets/Lab%20-%20Growing%20Datasets.ipynb
