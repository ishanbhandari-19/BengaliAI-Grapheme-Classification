# BengaliAI-Grapheme-Classification
Given the image of a handwritten Bengali grapheme, the challenge is to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

[Competition Link](https://www.kaggle.com/c/bengaliai-cv19)

## Approach
* Used a single Resnet18 Model backbone for all three labels, and used CrossEntropy Loss.
* Grapheme loss was given more weight, since the number of classes of grapheme were high when compared to the other two.
* 5 - fold stratified split training.

