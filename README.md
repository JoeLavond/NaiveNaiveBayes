# NaiveNaiveBayes
PyTorch semi-supervised implementation of Naive Bayes additionally assuming full feature independence.

### Model Derivations
See below for the derivation of the model.
$$
\begin{align}
    P(Y = c \mid X = x) &= \dfrac{P(Y = c \cap X = x)}{P(X = x)} \\
    &= \dfrac{P(Y = c) * P(X = x \mid Y = c)}{P(X = x)} \\
    &= P(Y = c) \left( \dfrac{\prod_i P(X_i = x_i \mid Y = c)}{P(X = x)} \right)  &&\textrm{Naive Bayes (Conditional Independence)} \\
    &= P(Y = c) \left( \dfrac{\prod_i P(X_i = x_i \mid Y = c)}{\prod_i P(X_i = x_i)} \right)  &&\textrm{Naive Naive Bayes (Independence)} \\
    &= P(Y = c) \left( \prod_i \left( \dfrac{P(X_i = x_i \mid Y = c)}{P(X_i = x_i)} \right) \right)
\end{align}
$$

Note the following implementation details if only using one-hot encoded vectors
1. All model coeficients can be quickly computed from weighted column averages
2. Predictions for new data points require only a cumulative product of masked model coeficients

Interpretability: Model coeficients are all very meaningful! \n
Notice that each term in the product indicated the ratio of the change in the belief $Y = c$ given that we now observe $X_i = x_i$.
$$ 
\begin{align}
    \dfrac{P(Y = c \mid X_i = x_i)}{P(Y = c)} &= P(Y = c \mid X_i = x_i) * \dfrac{1}{P(Y = c)} \\
    &= \dfrac{P(Y = c) * P(X_i = x_i \mid Y = c)}{P(X_i = x_i)} * \dfrac{1}{P(Y = c)}  &&\textrm{Bayes Theorem} \\
    &= \dfrac{P(Y = c)}{P(Y = c)} * \dfrac{P(X_i = x_i \mid Y = c)}{P(X_i = x_i)} \\
    &= \dfrac{P(X_i = x_i \mid Y = c)}{P(X_i = x_i)} 
\end{align}
$$

Comments:
1. Feature selection is a must. Tune either $p$, $k$, or $v$ in training. Many values in the product term close to 1, can cause significant variability from little to no reduction in bias. After feature selection, this model often performs significantly better.
2. For semi-supervised learning, model will predict unlabled data and be re-fit to the dataset using those predicted labels.
3. Although the independence assumption limits the complexity of relationships that this model can represent, we get huge benefits to interpretability, and this model is a great, fast starting points for understanding your data or basic feature selection.

