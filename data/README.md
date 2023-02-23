COSTRA 1.1: A Dataset of Complex Sentence Transformations and Comparisons
            Petra Barancikova 
            June 15, 2020
                  
This folder contains data set for testing of sentence embeddings in Czech. For 
detailed description of the dataset read the following article:
Barancikova P., Bojar O.: Costra 1.1: An Inquiry into Geometric Properties of Sentence Spaces,
Proceedings of the 23rd International Conference on Text, Speech and Dialogue (TSD 2020), 2020 

                       
1. Data
   1.1. Description
        The dataset is an extension of Costra 1.0 (Barancikova and Bojar, 2020), 
        which was extended in two rounds of manual annotation. For more detail 
        on Costra 1.0., see https://arxiv.org/abs/1912.01673
       

        Round 1: Extrapolations and interpolation were manually written by 7 annotators
                 for the following transformations: formal sentence, generalization,  
                 nonstandard sentence, opposite meaning, past and future.

        Round 2: Annotators were asked to compare pairs of sentences from the first 
                 round and Costra 1.0 based on their original transformation, i.e.,
                 which sentence is more general/in the past/nonstandard, etc.
                 
                 The annotators were given 4 options to compare sentences S1 and S2:
                 1) S1 is more general/formal/in the past/non-standard/... than S_2.
                 2) S1 is less general/formal/in the past/non-standard/... than S_2
                 3) S1 and S2 are too similar with respect to a selected criterion.
                 4) S1 and S2 are too dissimilar with respect to a selected criterion.

                 A pair of sentences may have multiple annotations. We present the 
                 relation only if it is an option with the majority of annotators' 
                 votes. Otherwise, it is discarded.

 
   1.2. Data Format
        The data file data.tsv contains 8 columns: 
        1) id: unique id for each sentence and type of transformation
        2) sentence id: unique id for each seed, from which the sentence was derived.
        3) transformation 
        4) sentence
        5) ids of more general/formal/in the past/non-standard/... sentences        
        6) ids of less general/formal/in the past/non-standard/... sentences
        7) ids of sentences too similar with respect to a selected criterion
        8) ids of sentences too dissimilar with respect to a selected criterion
                                                    
        Example:
        80	10	formal sentence	Čtyři mí příbuzní zesnuli.	83	79,82	78	
        80 - id of formal sentence "Čtyři mí příbuzní zesnuli."
        10 - sentence id of all sentences derived from the same seed sentence
        formal sentence - the type of transformation
        "Čtyři mí příbuzní zesnuli." - the sentence itself
        83 - a sentence with id 83 is more general than this sentence
        79,82 - sentences with ids 79,82 are less general than this sentence
        78 - a sentence with id 78 is too similar with regard to formalness
        "" - the last column is empty, no sentence was labeled as too dissimilar                 
        
         
2. License
   The data is made available under the terms of the Creative Commons Attribution 
   (CC-BY) license, version 4.0. You may use them for all purposes as long as the 
   authors are properly credited. 

3. Authors
   Petra Barancikova <barancikova@ufal.mff.cuni.cz>
   Ondrej Bojar  <bojar@ufal.mff.cuni.cz>,

   Charles University
   Faculty of Mathematics and Physics
   Institute of Formal and Applied Linguistics
   Malostranske nam 25
   118 00 Prague 1
   Czech Republic

References
Barancikova Petra, Bojar Ondrej: COSTRA 1.0: A Dataset of Complex Sentence Transformations. 
Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC 2020), 
Paris, France, ISBN 979-10-95546-34-4, pp. 3535–3541, 2020.
