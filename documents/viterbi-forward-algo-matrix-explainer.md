### From Nested Sums to Matrix Multiplication: The Intuition

Equation 3 represents the approximated likelihood function derived using a **rectangular quadrature rule** to discretize the continuous latent log-volatility process into $m$ distinct states.

$$L_{\text{approx}} = b^T \sum_{i_1=1}^m \dots \sum_{i_T=1}^m p(y_1 \mid y_0, h_1 = b^*_{i_1}) p(h_1 = b^*_{i_1}) \times \prod_{t=2}^T p(y_t \mid y_{t-1}, h_t = b^*_{i_t}) p(h_t = b^*_{i_t} \mid h_{t-1} = b^*_{i_{t-1}})$$

Evaluating Equation 3 directly by nesting $T$ individual summations requires calculating $m^T$ path combinations. If you have $m = 100$ grid points and $T = 1000$ daily observations, computing $100^{1000}$ operations is completely impossible.

To resolve this, the authors exploit the structure of a **Hidden Markov Model (HMM)**. Because the next hidden state $h_t$ only depends on the immediate previous state $h_{t-1}$ (the Markov property), we do not need to look at the entire history of the path all at once. We can calculate the probabilities step-by-step using dynamic programming (specifically, the **HMM Forward Algorithm**).

Matrix algebra perfectly encapsulates this step-by-step collapsing of history. Instead of evaluating all paths simultaneously, Equation 4 processes the data chronologically across time $t = 1, 2, \dots, T$, turning an exponential nightmare into a fast sequence of matrix multiplications.

---

### Breakdown of Components in Equation 4

Equation 4 rewrites the likelihood compactly as:

$$L_{\text{approx}} = \delta P(y_1) \Gamma P(y_2) \cdots \Gamma P(y_T) \mathbf{1}'$$

Here is the exact role of each matrix component:

1. 
**$\delta$ (Initial Probability Vector):** * **Dimensions:** $1 \times m$ (Row vector) 


* 
**Role:** It represents the starting probability distribution of the hidden log-volatility process at $t=1$. It is computed by evaluating the stationary continuous normal distribution at the $m$ discrete grid points and weighting them by the step size $b$.




2. 
**$P(y_t)$ (Emission / Observation Matrix):** * **Dimensions:** $m \times m$ (Diagonal matrix) 


* 
**Role:** For a given day $t$, this matrix answers the question: *“How likely is it that we observed the return $y_t$, assuming the hidden volatility is currently sitting at state $1, 2, \dots,$ or $m$?”*  Because it is a diagonal matrix, multiplying by it simply scales each state's incoming probability by its corresponding observation likelihood.




3. 
**$\Gamma$ (Transition Probability Matrix):** * **Dimensions:** $m \times m$ (Square matrix) 


* 
**Role:** This matrix dictates the dynamics of the hidden state process. The element $\gamma_{ij}$ represents the probability of transitioning from hidden state $i$ at time $t-1$ to hidden state $j$ at time $t$.




4. 
**$\mathbf{1}'$ (Summing Vector):** * **Dimensions:** $m \times 1$ (Column vector of ones) 


* **Role:** At the very end of the time series ($t=T$), we are left with a row vector containing the total probabilities of finishing the entire observed sequence in each of the $m$ hidden states. Multiplying by a column vector of ones sums up these elements to yield a single scalar value: the total marginal likelihood of the data.





---

### The Intuitive Role of Matrix Multiplication

To see how the math moves sequentially, follow the calculation from left to right:

* **Step 1: Start and Observe $y_1 \rightarrow [\delta P(y_1)]$.**
Multiplying the initial row vector $\delta$ by the diagonal matrix $P(y_1)$ gives a new $1 \times m$ row vector. Each element $i_1$ in this vector represents the joint probability: *"The system started in state $i_1$ **AND** we observed return $y_1$."*
* **Step 2: Move to day 2 $\rightarrow [(\delta P(y_1)) \Gamma]$.**
When you multiply this row vector by the transition matrix $\Gamma$, the inner product mechanics of matrix multiplication automatically perform a summation: it multiplies the probability of having been in state $i_1$ by the probability of moving from $i_1 \rightarrow i_2$, and sums this over all $m$ possible choices of $i_1$. This collapses the historical paths and leaves you with a $1 \times m$ row vector of probabilities for arriving at each state $i_2$ on day 2.
* **Step 3: Observe $y_2 \rightarrow [(\delta P(y_1) \Gamma) P(y_2)]$.**
We multiply by the diagonal matrix $P(y_2)$, updating our day 2 arrival probabilities with the likelihood of actually seeing the return data $y_2$ given those states.

This cycle—**Transition ($\Gamma$) then Observe ($P(y_t)$)**—repeats recursively for every time step up to $T$. By utilizing matrix multiplication to sum out the previous day's dependencies at every step, the algorithm keeps the calculation size strictly bounded to a computational cost of $O(T \cdot m^2)$, making parameter estimation instantaneous.