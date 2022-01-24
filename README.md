# Word2Vec

## Architecture Design

### Neural Probabilistic Language Model Architecture:



*   Input layer
      
      * N previous words are encoded using **1-of-V coding**, where V is vocabulary size
*   Projection layer(Embedding layer)

      * The input layer is then projected to a projection layer P that has dimension N x D, N inputs are active at any given time
*   Hidden layer

      * NNLM architecture becomes complex for computation between the projection and the hidden layer, as values in the projection layer are dense
      *  If N = 10, the size of the projection layer (P) might be 500 to 2000, while the hidden layer size H is typically 500 to 1000 units.
      Thus complexity = N x D + N x D x H + H x V

*   Output layer

    *
    

## New Log-linear Models:


*   Proposed 2 new model architectures for learning distributed representations of words that try to minimize computational complexity
*   In the previous section, the complexity is caused by the non-linear hidden layer in the model, while this non-linearity makes the neural net special
*    They explored much simpler model, which trained on large data can perform better result
    *   Simple neural network i.e with hidden layer trained in two steps:
        1. continuous word vectors are learned using simple model
        2. Then N-gram NNLM is trained on top of these distributed representation of words.




## Continuous Bag-of-Words Model:


*   similar to feedforward NNLM, where the non-linear hidden layer is removed and the projection layer is shared for all words ( not just the projection matrix)
*   Called continuous bag of words because the order of words in the history does not influence the projection.

*    found best performance using history and future words
*    Training complexity: 
        Q = N x D + D x log_2(V)


* Input layer
* Projection layer (embedding layer) -> get embedding of target and context words
* Output layer -> dot product of target_embedding and context_embedding
    * takes care on dimensions of the matrix
    *  why bias is not present ?

Reference:

https://arxiv.org/pdf/1301.3781.pdf

http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

https://www.analyticsvidhya.com/blog/2021/06/part-5-step-by-step-guide-to-master-nlp-text-vectorization-approaches/

https://github.com/chao-ji/tf-word2vec


https://github.com/mchablani/deep-learning/blob/master/embeddings/Skip-Gram_word2vec.ipynb



## Skip-gram model



*   Predict neighboring context given a word
*   Objective: maximize the average log probability:
    
    * (![image](https://user-images.githubusercontent.com/21074651/150743118-22a958f7-19fa-4b5d-93cb-3dd6aad4cd13.png)
)

    *  c is the size of the training context

* Skip-gram defines the probability using softmax function:

(![image](https://user-images.githubusercontent.com/21074651/150743150-1a6aa323-123b-42f1-a7e9-35089b8d8383.png)
)

    *  W is the number of words in vocabulary
    *  Output layer gives the probability of words in the entire vocabulary
    *    Computing cost of probability over entire vocab can be inefficient, so other approximation methods are developed
    i. Hierarchical Softmax
        * replace softmax with sigmoid
        * binary tree
    ii. Negative Sampling
        *  understand negative sampling objective
        * the task is to distinguish the target word w_o from draws from the noise distribution using logistic regression(Hinge loss, binary classification)
        * Paper: noise distribution is a free parameter
    iii. Subsampling of frequent words
        *   most frequent words(a, the, an, etc) provides less information and their representation won't change significantly after training on several million examples
        *  to counter the imbalance between the rare and frequent words, a simple subsampling approach is used
        * heuristic sub sampling technique is used in paper


