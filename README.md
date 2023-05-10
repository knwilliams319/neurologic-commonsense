# neurologic-commonsense
Group 8's Research Project for the S23 edition of CS397 Seminar in Statistical Language Modeling with Professor David Demeter. 
Participants: Kyle Williams, Arya Bulusu, Yemi Kelani, Rodney Reichart


## Project Description
Our goal is to enhance how LLMs generate language by leveraging existing knowledge bases or knowledge graphs. By learning how to model language overall, LLMs are great at a number of different general tasks, and can even memorize human knowledge during their training. However, with respect to a specific knowledge base, embedding that knowledge into a model is an interesting problem. Transformer architectures necessitate large training corpora; most knowledge bases are not large enough to train a model from scratch. Even if the knowledge base is large, incorporating it into the training phase risks making the model worse at general-purpose language modeling in an attempt to memorize the facts from the base. But, there is no guarantee the model memorizes the facts in the first place. Further, the details of how knowledge from the base is best represented in natural language is a complicated question in its own right. A logical approach might be to fine-tune an existing LLM, but again, this may not always be feasible, especially if the knowledge base is small. We reason that Neurologic Decoding can be leveraged in a way that allows any existing LLM to connect to any existing knowledge base without further modification of the model or knowledge base. We wish to investigate as many different extensions of this text-generation method as possible within the scope of this project. 

## Data Sources
Because we’re attempting to supplement models with information from a Knowledge Base, our evaluation dataset will need to be tied to the knowledge base in some manner. Luckily, the CommonSenseQA dataset is covered in class, and was created from the ConceptNet Knowledge Base. We want to use this pair of sources to see if we can improve upon the benchmarks shown in class. If time permits, we would also like to validate our method on a smaller (Knowledge Base, Evaluation Dataset) pair. We believe our approach will significantly improve transformers’ abilities to perform on datasets too small for fine-tuning. 

## Our Model / Task
Our goal is to alter Neurologic Decoding in a way that allows large pretrained models to connect to a knowledge base without any extra effort or fine tuning. To this end, we will supplement input sequences with destructured knowledge queried from a knowledge graph (ConceptNet), and fine-tune a pretrained language model on an extractive question-answering task. Furthermore, we will use a generative LLM like T5 or GPT-2, then change the way it decodes and generates sentences using our version of Neurologic Decoding. Neurologic Decoding involves listing tokens that should (or should not) be included in the generated text via CNF. Our extension will primarily involve how the CNF clauses are filled with items from the knowledge base. Neurologic Decoding incorporates various hyperparameters to tune the generation, and additional parameters may help us formalize how deep our graph search should be, or with what proportion the model should favor using concepts from the base as opposed to its own generated text. 

We’re framing an open generative task as an extractive question-answering task with the hope that the model can generate relevant text having extracted the answer from the base context and/or supplemental knowledge. 

## Evaluation and Benchmarks
In the CommonSenseQA paper, the researchers evaluate models by calculating their overall accuracy. This is possible because each question has multiple-choice answers. We also believe that accuracy is important to validating our approach. Additionally, because the paper on CommonSenceQA evaluates performance on the dataset using accuracy, it will enable us to do an apples to apples comparison. Because our focus is on improving text generation, we want our model to answer the question in its own words. Thus, the model will be “accurate” if it uses the correct multiple-choice option in its generated sentence. Accuracy doesn’t tell the whole story, though. We don’t want our models to be correct only because they were fed the right knowledge from the base. We also want them to form contextually-relevant answers. Thus, we will also incorporate some kind of similarity score over the words in the question (e.g. BLEU or BERTScore). In regards to benchmarks, we will apply our Neurologic-Decoding variant to an LLM like T5. We will compare the results of our decoding variant against the zero-shot and fine-tuned performances of the same LLM.

## Acknowledgements
This work includes data from ConceptNet 5, which was compiled by the
Commonsense Computing Initiative. ConceptNet 5 is freely available under
the Creative Commons Attribution-ShareAlike license (CC BY SA 3.0) from
http://conceptnet.io.

The included data was created by contributors to Commonsense Computing
projects, contributors to Wikimedia projects, DBPedia, OpenCyc, Games
with a Purpose, Princeton University's WordNet, Francis Bond's Open
Multilingual WordNet, and Jim Breen's JMDict.
