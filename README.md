# Propaganda-Detection-at-SemEval-2020

The project describes a fast solution to propaganda detection at SemEval-2020 Task 11, based onfeature adjustment. We use per-token vectorization of features and a simple Logistic Regression Classifier to quickly test different hypotheses about our data. The result of our system at SemEval2020 Task 11 is F-score=0.37.

The file main.py contains the classfier that uses three .csv files with vectors. Due to limits in file size, these tables are parts of what we used at the competition.

The file get_data.py contains the feature vector processor to produce vectors from Propaganda project articles. The code is sensitive to the format of the data set: file numbering and labels.

The rest are support files for Gensim Word2vec and our prorpocessor based on the Roget's thesaurus.

More details: https://www.researchgate.net/publication/343849942_UTMN_at_SemEval-2020_Task_11_A_Kitchen_Solution_to_Automatic_Propaganda_Detection

Cite:

@inproceedings{mikhalkova2020utmn,

  title={UTMN at SemEval-2020 Task 11: A Kitchen Solution to Automatic Propaganda Detection},
  
  author={Mikhalkova, Elena and Ganzherli, Nadezhda and Glazkova, Anna and Bidulya, Yuliya},
  
  booktitle={Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  
  pages={1858--1864},
  
  year={2020}
  
}
