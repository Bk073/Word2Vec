# Word2Vec

## Architecture Design

### Neural Probabilistic Language Model Architecture:



*   Input layer
      
      * N previous words are encoded using **1-of-V coding**, where V is vocabulary size
*   Projection layer

      * The input layer is then projected to a projection layer P that has dimension N x D, N inputs are active at any given time
*   Hidden layer

      * NNLM architecture becomes complex for computation between the projection and the hidden layer(non-linear), as values in the projection layer are dense
      *  If N = 10, the size of the projection layer (P) might be 500 to 2000, while the hidden layer size H is typically 500 to 1000 units.
      Thus complexity = N x D + N x D x H + H x V

*   Output layer

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


## Skip-gram model



*   Predict neighboring context given a word
*   Objective: maximize the average log probability:
    
    * ![image](https://user-images.githubusercontent.com/21074651/150743118-22a958f7-19fa-4b5d-93cb-3dd6aad4cd13.png)


    *  c is the size of the training context

* Skip-gram defines the probability using softmax function:
    *   to calculate this softmax over large vocabulary is ineffective

![image](https://user-images.githubusercontent.com/21074651/150743150-1a6aa323-123b-42f1-a7e9-35089b8d8383.png)


    *  W is the number of words in vocabulary
    *  Output layer gives the probability of words in the entire vocabulary
    *    Computing cost of probability over entire vocab can be inefficient, so other approximation methods are developed
    i. Hierarchical Softmax
        * computationally efficient approximation of the full softmax
        * replace the standard softmax with sigmoid in standard Skip-gram model
    ii. Negative Sampling
        *  the objective of using negative sampling is that a good model should be able to differentiate data from noise
        * the task is to distinguish the target word w_o from the noise distribution using logistic regression(Hinge loss, binary classification)
        * Paper: noise distribution is a free parameter
    iii. Subsampling of frequent words
        *   most frequent words(a, the, an, etc) provides less information and their representation won't change significantly after training on several million examples
        *  to counter the imbalance between the rare and frequent words, a simple subsampling approach is used
        * heuristic sub sampling technique is used in paper


### Results:

Sampled 500 vocaublary words

![image](https://user-images.githubusercontent.com/21074651/150744196-93d18a56-8b1f-430c-ae5c-93fe6489aa8a.png)

![image](https://user-images.githubusercontent.com/21074651/150744526-3b096b53-c054-46b1-a2d6-3a8d59b41899.png)

## Reference:

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- A Neural Probabilistic Language Model



## TODO:

*   Error analysis
      * Currently the results are not good.   
