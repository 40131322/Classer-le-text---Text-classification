# Classer-le-text---Text-classification

Description
L’objectif est de concevoir un algorithme d’apprentissage automatique capable de trier automatiquement de courts documents textuels dans un ensemble de catégories prédéterminées. Vous recevez des vecteurs de comptage de mots (termes) par document comme caractéristiques, où chaque index représente le nombre de fois qu’un terme donné est présent dans le document. Parallèlement à ces vecteurs, vous recevez également une carte de vocabulaire qui associe chaque index à un terme. Votre objectif est d’exploiter cette matrice de comptage de termes pour résoudre une tâche de classification de texte binaire.

The goal is to design a machine learning algorithm that can automatically sort short text documents into a pre-determined set of categories. You are given document-wise word (term) count vectors as features, where each index is the number of times that a given term is present in the document. Alongside these vectors, you are also given a vocabulary map that corresponds to each index to a term. Your goal is to leverage this term count matrix to solve a binary text classification task.

Evaluation
Pour évaluer votre modèle, vous devez créer un fichier csv avec les prédictions pour les exemples dans data_test.pkl. Le format est donné en-dessous. La métrique utilisée est la score macro f1.

In order to evaluate your model, you need to create a csv file with predictions for the examples in data_test.npy. The formatting is given below. We use the macro f1 score metric.

ID,label
0,1
1,1
2,0
3,0
4,1
5,...
