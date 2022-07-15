# Statistics_and_Probability_Project
# Machine Learning Week 1 Quiz 1 (Introduction) Stanford Coursera

Question 1
----------
A computer program is said to learn from experience E with

respect to some task T and some performance measure P if its

performance on T, as measured by P, improves with experience E.

Suppose we feed a learning algorithm a lot of historical weather

data, and have it learn to predict weather. What would be a

reasonable choice for P?

* The probability of it correctly predicting a future date's weather.

* None of these.

* The weather prediction task.

* The process of the algorithm examining a large amount of historical weather data.

Answer: </br>

The probability of it correctly predicting a future date's weather.


Question 2
----------
Suppose you are working on weather prediction, and use a

learning algorithm to predict tomorrow's temperature (in

degrees Centigrade/Fahrenheit).

Would you treat this as a classification or a regression problem?

* Regression

* Classification

Answer: </br>

Regression

Question 3
----------
Suppose you are working on stock market prediction. You would like to predict whether or not a certain company will declare bankruptcy within the next 7 days (by training on data of similar companies that had previously been at risk of bankruptcy). Would you treat this as a classification or a regression problem?

* Regression

* Classification

Answer: </br>

Classification

Question 4
----------
Some of the problems below are best addressed using a supervised

learning algorithm, and the others with an unsupervised

learning algorithm. Which of the following would you apply

supervised learning to? (Select all that apply.) In each case, assume some appropriate

dataset is available for your algorithm to learn from.

* Examine a web page, and classify whether the content on the web page should be considered "child friendly" (e.g., non-pornographic, etc.) or "adult."

* Given data on how 1000 medical patients respond to an experimental drug (such as effectiveness of the treatment, side effects, etc.), discover whether there are different categories or "types" of patients in terms of how they respond to the drug, and if so what these categories are.

* In farming, given data on crop yields over the last 50 years, learn to predict next year's crop yields.

* Given a large dataset of medical records from patients suffering from heart disease, try to learn whether there might be different clusters of such patients for which we might tailor separate treatments.

Answer: </br>

True or False | Statement 
--- | ---
True | Examine a web page, and classify whether the content on the web page should be considered "child friendly" (e.g., non-pornographic, etc.) or "adult."
True | In farming, given data on crop yields over the last 50 years, learn to predict next year's crop yields.

Question 5
----------
Which of these is a reasonable definition of machine learning?

* Machine learning is the field of allowing robots to act intelligently.

* Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.

* Machine learning learns from labeled data.

* Machine learning is the science of programming computers.

Answer: </br>

Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.
# Machine Learning Week 1 Quiz 2 (Linear Regression with One Variable) Stanford Coursera

Question 1
----------
Consider the problem of predicting how well a student does in her second year of college/university, given how well she did in her first year.

Specifically, let x be equal to the number of "A" grades (including A-. A and A+ grades) that a student receives in their first year of college (freshmen year). We would like to predict the value of y, which we define as the number of "A" grades they get in their second year (sophomore year).

Here each row is one training example. Recall that in linear regression, our hypothesis is h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x, and we use m to denote the number of training examples.

x | y
--- | ---
5 | 4
3 | 4
0 | 1
4 | 3

For the training set given above (note that this training set may also be referenced in other questions in this quiz), what is the value of m? In the box below, please enter your answer (which should be a number between 0 and 10).

Answer: </br>
4

Question 2
----------
Consider the following training set of m=4 training examples:

x | y 
--- | --- 
1 | 0.5
2 | 1
4 | 2
0 | 0

Consider the linear regression model h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x. What are the values of θ<sub>0</sub> and θ<sub>1</sub> that you would expect to obtain upon running gradient descent on this model? (Linear regression will be able to fit this data perfectly.)

* θ<sub>0</sub>=0.5,θ<sub>1</sub>=0

* θ<sub>0</sub>=0.5,θ<sub>1</sub>=0.5

* θ<sub>0</sub>=1,θ<sub>1</sub>=0.5

* θ<sub>0</sub>=0,θ<sub>1</sub>=0.5

* θ<sub>0</sub>=1,θ<sub>1</sub>=1

Answer: </br>
θ<sub>0</sub>=0,θ<sub>1</sub>=0.5 </br>

As J(θ<sub>0</sub>,θ<sub>1</sub>)=0, y = h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x. Using any two values in the table, solve for θ<sub>0</sub>, θ<sub>1</sub>. 

If you don't know how to do this, please see the following video: [Solving system of linear equations](https://www.youtube.com/watch?v=AqIrdW2-K6k&t=2s)

Question 3
----------
Suppose we set θ<sub>0</sub>=−1,θ<sub>1</sub>=0.5. What is h<sub>θ</sub>(4)?

Answer: </br>

Setting x = 4, we have h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x = -1 + (0.5)(4) = <b>1</b>

Question 4
----------
Let f be some function so that

f(θ<sub>0</sub>,θ<sub>1</sub>) outputs a number. For this problem,

f is some arbitrary/unknown smooth function (not necessarily the

cost function of linear regression, so f may have local optima).

Suppose we use gradient descent to try to minimize f(θ<sub>0</sub>,θ<sub>1</sub>)
as a function of θ<sub>0</sub> and θ<sub>1</sub>. Which of the

following statements are true? (Check all that apply.)

* Even if the learning rate α is very large, every iteration of gradient descent will decrease the value of f(θ<sub>0</sub>,θ<sub>1</sub>).

* If the learning rate is too small, then gradient descent may take a very long time to converge.

* If θ<sub>0</sub> and θ<sub>1</sub> are initialized at a local minimum, then one iteration will not change their values.

* If θ<sub>0</sub> and θ<sub>1</sub> are initialized so that θ<sub>0</sub>=θ<sub>1</sub>, then by symmetry (because we do simultaneous updates to the two parameters), after one iteration of gradient descent, we will still have θ<sub>0</sub>=θ<sub>1</sub>.

Answers: </br>

True or False | Statement | Explanation 
--- | --- | ---
True | If the learning rate is too small, then gradient descent may take a very long time to converge. | If the learning rate is small, gradient descent ends up taking an extremely small step on each iteration, and therefor can take a long time to converge
True | If θ<sub>0</sub> and θ<sub>1</sub> are initialized at a local minimum, then one iteration will not change their values. | At a local minimum, the derivative (gradient) is zero, so gradient descent will not change the parameters. 
False | Even if the learning rate α is very large, every iteration of gradient descent will decrease the value of f(θ<sub>0</sub>,θ<sub>1</sub>). | If the learning rate is too large, one step of gradient descent can actually vastly "overshoot" and actually increase the value of f(θ<sub>0</sub>,θ<sub>1</sub>). 
False | If θ<sub>0</sub> and θ<sub>1</sub> are initialized so that θ<sub>0</sub>=θ<sub>1</sub>, then by symmetry (because we do simultaneous updates to the two parameters), after one iteration of gradient descent, we will still have θ<sub>0</sub>=θ<sub>1</sub>. | The updates to θ<sub>0</sub> and θ<sub>1</sub> are different (even though we're doing simulaneous updates), so there's no particular reason to update them to be same after one iteration of gradient descent. 

Other Options: </br>

True or False | Statement | Explanation 
--- | --- | ---
True | If the first few iterations of gradient descent cause f(θ<sub>0</sub>,θ<sub>1</sub>) to increase rather than decrease, then the most likely cause is that we have set the learning rate to too large a value  | if  alpha were small enough, then gradient descent should always successfully take a tiny small downhill and decrease f(θ<sub>0</sub>,θ<sub>1</sub>) at least a little bit. If gradient descent instead increases the objective value, that means alpha is too large (or you have a bug in your code!).
False | No matter how θ<sub>0</sub> and θ<sub>1</sub> are initialized, so long as learning rate is sufficiently small, we can safely expect gradient descent to converge to the same solution | This is not true, depending on the initial condition, gradient descent may end up at different local optima. 
False | Setting the learning rate to be very small is not harmful, and can only speed up the convergence of gradient descent. | If the learning rate is small, gradient descent ends up taking an extremely small step on each iteration, so this would actually slow down (rather than speed up) the convergence of the algorithm. 

Question 5
----------
Suppose that for some linear regression problem (say, predicting housing prices as in the lecture), we have some training set, and for our training set we managed to find some θ<sub>0</sub>, θ<sub>1</sub> such that J(θ<sub>0</sub>,θ<sub>1</sub>)=0.

Which of the statements below must then be true? (Check all that apply.)

* For this to be true, we must have y<sup>(i)</sup>=0 for every value of i=1,2,…,m.

* Gradient descent is likely to get stuck at a local minimum and fail to find the global minimum.

* For this to be true, we must have θ<sub>0</sub>=0 and θ<sub>1</sub>=0 so that h<sub>θ</sub>(x)=0

* Our training set can be fit perfectly by a straight line, i.e., all of our training examples lie perfectly on some straight line.

Answers: </br>

True or False | Statement | Explanation 
--- | --- | ---
False | For this to be true, we must have y<sup>(i)</sup>=0 for every value of i=1,2,…,m. | So long as all of our training examples lie on a straight line, we will be able to find θ<sub>0</sub> and θ<sub>1</sub>) so that J(θ<sub>0</sub>,θ<sub>1</sub>)=0. It is not necessary that y<sup>(i)</sup> for all our examples. 
False | Gradient descent is likely to get stuck at a local minimum and fail to find the global minimum. | none
False | For this to be true, we must have θ<sub>0</sub>=0 and θ<sub>1</sub>=0 so that h<sub>θ</sub>(x)=0 | If J(θ<sub>0</sub>,θ<sub>1</sub>)=0 that means the line defined by the equation "y = θ<sub>0</sub> + θ<sub>1</sub>x" perfectly fits all of our data. There's no particular reason to expect that the values of θ<sub>0</sub> and θ<sub>1</sub> that achieve this are both 0 (unless y<sup>(i)</sup>=0 for all of our training examples).
True | Our training set can be fit perfectly by a straight line, i.e., all of our training examples lie perfectly on some straight line. | None

Other Options: </br>

True or False | Statement | Explanation 
--- | --- | ---
False | We can perfectly predict the value of y even for new examples that we have not yet seen. (e.g., we can perfectly predict prices of even new houses that we have not yet seen.)  | None
False | This is not possible: By the definition of J(θ<sub>0</sub>,θ<sub>1</sub>), it is not possible for there to exist θ<sub>0</sub> and θ<sub>1</sub> so that J(θ<sub>0</sub>,θ<sub>1</sub>)=0 | None
True | For these values of θ<sub>0</sub> and θ<sub>1</sub> that satisfy J(θ<sub>0</sub>,θ<sub>1</sub>)=0, we have that h<sub>θ</sub>(x<sup>(i)</sup>)=y<sup>(i)</sup> for every training example (x<sup>(i)</sup>,y<sup>(i)</sup>) | None

# Machine Learning Week 1 Quiz 3 (Linear Algebra) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/quiz3q1.png)

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/quiz3q2.png)

Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/quiz3q3.png)

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/quiz3q4.png)

Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/quiz3q5.png)

# Machine Learning Week 2 Quiz 1 (Linear Regression with Multiple Variables) Stanford Coursera

Question 1
----------
Suppose m=4 students have taken some class, and the class had a midterm exam and a final exam. You have collected a dataset of their scores on the two exams, which is as follows:

Midterm Exam | (midterm exam)<sup>2</sup> | Final Exam 
--- | --- | ---
89 | 7921 | 96
72 | 5184 | 74
94 | 8836 | 87
69 | 4761 | 78

You'd like to use polynomial regression to predict a student's final exam score from their midterm exam score. 
Concretely, suppose you want to fit a model of the form hθ(x)=θ<sub>0</sub>+θ<sub>1</sub>x<sub>1</sub>+θ<sub>2</sub>x<sub>2</sub>, 
where x<sub>1</sub> is the midterm score and x<sub>2</sub> is (midterm score)<sup>2</sup>. 
Further, you plan to use both feature scaling (dividing by the "max-min", or range, of a feature) and mean normalization.

What is the normalized feature x<sub>2</sub><sup>(4)</sup>? (Hint: midterm = 69, final = 78 is training example 4.) 
Please round off your answer to two decimal places and enter in the text box below.

Answer: </br>

The mean of x<sub>2</sub> is 6675.5 and the range is 8836 - 4761 is 4075. </br>

x<sub>2</sub><sup>(4)</sup> = (4761 - 6675.5) / 4075 = <b>-0.47</b>

Question 2
----------
You run gradient descent for 15 iterations
with α=0.3 and compute
J(θ) after each iteration. You find that the
value of J(θ) <b>decreases</b> quickly then levels
off. Based on this, which of the following conclusions seems
most plausible?

* Rather than use the current value of α, it'd be more promising to try a larger value of α (say α=1.0).

* Rather than use the current value of α, it'd be more promising to try a smaller value of α (say α=0.1).

* α=0.3 is an effective choice of learning rate.

Answer: </br>

Answer | Explanation
--- | ---
α=0.3 is an effective choice of learning rate. | We want gradient descent to quickly converge to the minimum, so the current setting of α seems to be good

Question 3
----------
Suppose you have m=14 training examples with n=3 features (excluding the additional all-ones feature for the intercept term, which you should add). 
The normal equation is θ=(X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>y. 
For the given values of m and n, what are the dimensions of θ, X, and y in this equation?

* X is 14×3, y is 14×1, θ is 3×3

* X is 14×4, y is 14×4, θ is 4×4

* X is 14×4, y is 14×1, θ is 4×1

* X is 14×3, y is 14×1, θ is 3×1

Answer | Explanation
--- | ---
X is 14×4, y is 14×1, θ is 4×1 | X has m rows and n + 1 columns (+1 because of the x<sub>0</sub>=1 term. y is an m-vector. θ is an (n+1)-vector.

Question 4
----------
Suppose you have a dataset with m=50 examples and n=200000 features for each example. You want to use multivariate linear regression to fit the parameters θ to our data. Should you prefer gradient descent or the normal equation?

* Gradient descent, since (X<sup>T</sup>X)<sup>−1</sup> will be very slow to compute in the normal equation.

* Gradient descent, since it will always converge to the optimal θ.

* The normal equation, since it provides an efficient way to directly find the solution.

* The normal equation, since gradient descent might be unable to find the optimal θ.

Answer | Explanation
--- | ---
Gradient descent, since (X<sup>T</sup>X)<sup>−1</sup> will be very slow to compute in the normal equation. | With n = 200000 features, you have to invert a 200001 x 200001 matrix to compute the normal equation. Inverting such a large matrix is computationally expensive, so gradient descent is a good choice. 

Question 5
----------
Which of the following are reasons for using feature scaling?

* It speeds up solving for θ using the normal equation.

* It prevents the matrix X<sup>T</sup>X (used in the normal equation) from being non-invertable (singular/degenerate).

* It is necessary to prevent gradient descent from getting stuck in local optima.

* It speeds up gradient descent by making it require fewer iterations to get to a good solution.

True or False | Statement | Explanation 
--- | --- | ---
False | It speeds up solving for θ using the normal equation. | The magnitude of the feature values are insignificant in terms of computational cost. 
False | It prevents the matrix X<sup>T</sup>X (used in the normal equation) from being non-invertable (singular/degenerate). | none
False | It is necessary to prevent gradient descent from getting stuck in local optima. | The cost function J(θ) for linear regression has no local optima. 
True | It speeds up gradient descent by making it require fewer iterations to get to a good solution. | Feature scaling speeds up gradient descent by avoiding many extra iterations that are required when one or more features take on much larger values than the rest. 

# Week 2 | V. Octave Tutorial

## Question 1


Suppose I first execute the following Octave commands:

	A = [1 2; 3 4; 5 6];
	B = [1 2 3; 4 5 6];

Which of the following are then valid Octave commands? Check all that apply. (Hint: A' denotes the transpose of A.)

### Answer

* **C = B' + A;**
 
* C = A' * B;
 
* C = B + A;
 
* **C = A * B;**


---

## Question 2
![image](http://frog.isima.fr/cgi-bin/bruno/tex2png--10.cgi?A = \begin{bmatrix}  16 &2&3&13 \\\\5&11&10&8\\\\9&7&6&12\\\\4&14&15&1 \end{bmatrix})

Which of the following indexing expressions gives

![image](http://frog.isima.fr/cgi-bin/bruno/tex2png--10.cgi?B = \begin{bmatrix}  16 &2 \\\\5&11\\\\9&7\\\\4&14 \end{bmatrix})
? Check all that apply.

### Answer

* B = A(:, 0:2);
 
* **B = A(:, 1:2);**
 
* **B = A(1:4, 1:2);**
 
* B = A(0:4, 0:2);

---

## Question 3
Let A be a 10x10 matrix and x be a 10-element vector. Your friend wants to compute the product Ax and writes the following code:

	v = zeros(10, 1);
	for i = 1:10
	  for j = 1:10 
	    v(i) = v(i) + A(i, j) * x(j);
	  end
	end
	
	
How would you vectorize this code to run without any for loops? Check all that apply.

### Answer

* **v = A * x;**
 
* v = sum (A * x);
 
* v = A .* x;
 
* v = Ax;


---

## Question 4
Say you have two column vectors v and w, each with 7 elements (i.e., they have dimensions 7x1). Consider the following code:

	z = 0;
	for i = 1:7
	  z = z + v(i) * w(i);
	end

Which of the following vectorizations correctly compute z? Check all that apply.

### Answer

* **z = v' * w;**
 
* z = v * w;
 
* **z = sum (v .* w);** 
 
* z = v .* w;


---

## Question 5
In Octave, many functions work on single numbers, vectors, and matrices. For example, the sin function when applied to a matrix will return a new matrix with the sin of each element. But you have to be careful, as certain functions have different behavior. Suppose you have an 7x7 matrix X. You want to compute the log of every element, the square of every element, add 1 to every element, and divide every element by 4. You will store the results in four matrices, A,B,C,D. One way to do so is the following code:

	for i = 1:7
	  for j = 1:7
    	A(i, j) = log (X(i, j));
	    B(i, j) = X(i, j) ^ 2;
    	C(i, j) = X(i, j) + 1;
	    D(i, j) = X(i, j) / 4;
	  end
	end
	
Which of the following correctly compute A,B,C, or D? Check all that apply.

### Answer

* **B = X .^ 2;**
 
* **C = X + 1;**
 
* **D = X / 4;**
 
* B = X ^ 2;

# Machine Learning Week 3 Quiz 1 (Logistic Regression) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/logisticq1.PNG)

Answer | Explanation
--- | ---
Our estimate for P(y=0\|x;θ) is 0.8. | P(y=0\|x;θ) = 1 - P(y = 1\| x; θ); the former is 1 - 0.2 = 0.8
Our estimate for P(y=1\|x;θ) is 0.2. | h<sub>θ</sub>(x) = 0.2

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/logisticQ2part2answers.png)

Answer | Explanation
--- | ---
J(θ) will be a convex function, so gradient descent should converge to the global minimum. | none
Adding polynomial features (e.g., instead using h<sub>θ</sub>(x) = g(θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x</sub>2 + θ<sub>3</sub>x<sup>2</sup> + θ<sub>4</sub>x<sup>1</sup>x<sup>2</sup> + θ<sub>5</sub>x<sup>2</sup> )) could increase how well we can fit the training data | Adding new features can only improve the fit on the training set: since setting θ<sub>3</sub> = θ<sub>4</sub> = θ<sub>5</sub> = 0 makes the hypothesis the same as the original one, gradient descent will use those features (by making the corresponding non-zero) only if doing so improves the training set fit

Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/logisticq3.PNG)

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/logisticq4.PNG)

Answer | Explanation
--- | ---
The cost function J(θ) for logistic regression trained with examples is always greater than or equal to zero. | The cost for any example x<sup>(i)</sup> is always ≥ 0 since it is the negative log of a quantity less than one. The cost function J(θ) is a summation over the cost for each eample, so the cost function itself must be greater than or equal to zero.
The sigmoid function is never greater than one | none


Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/logisticQ5part1.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/logisticQ5part2.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/logisticQ5answer.jpg)

# Machine Learning Week 3 Quiz 2 (Regularization) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ1.png)

True or False | Statement | Explanation 
--- | --- | ---
False | Adding many new features to the model helps prevent overfitting on the training set. | Adding many new features gives us more expressive models which are able to better fit our training set. If too many new features are added, this can lead to overfitting of the training set.
False | Introducing regularization to the model always results in equal or better performance on examples <b> not </b> in the training set. | If we introduce too much regularization, we can underfit the training set and this can lead to worse performance even for examples not in the training set.
False | Introducing regularization to the model always results in equal or better performance on the training set. | If we introduce too much regularization, we can underfit the training set and have worse performance on the training set.
True | Adding a new feature to the model always results in equal or better performance on the training set | Adding many new features gives us more expressive models which are able to better fit our training set. If too many new features are added, this can lead to overfitting of the training set.

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ2.png)

Answer | Explanation 
--- | --- 
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ2imageAnswer.png" alt="Answer Image"> | Adding many new features to the model helps prevent overfitting on the training set. 

Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ3.png)

True or False | Statement | Explanation 
--- | --- | ---
False | Using a very large value λ cannot hurt the performance of your hypothesis; the only reason we do not set to be too large is to avoid numerical problems. | Using a very large value of λ can lead to underfitting of the training set.
False | Because regularization causes J(θ) to no longer be convex, gradient descent may not always converge to the global minimum (when λ > 0, and when using an appropriate learning rate α). | Regularized logistic regression and regularized linear regression are both convex, and thus gradient descent will still converge to the global minimum.
True | Using too large a value of λ can cause your hypothesis to underfit the data.| A large value of results in a large λ regularization penalty and thus a strong preference for simpler models which can underfit the data.
False | Because logistic regression outputs values 0 <= h<sub>0</sub> <= 1, its range of output values can only be "shrunk" slighly by regularization anyway, so regularization is generally not helpful for it.  | None needed

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ4p1.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ4p2.png)

Answer | Explanation 
--- | --- 
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationq4imageAnswer.png" alt="Answer Image"> | The hypothesis follows the data points very closely and is highly complicated, indicating that it is overfitting the training set

Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ5p1.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationQ5p2.png)

Answer | Explanation 
--- | --- 
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/regularizationq5imageAnswer.png" alt="Answer Image"> | The hypothesis does not predict many data points well, and is thus underfitting the training set.

# Statistical Inference Quiz 3 (JHU) Coursera

Question 1
----------
In a population of interest, a sample of 9 men yielded a sample average brain volume of 1,100cc and a standard deviation of 30cc. What is a 95% Student's T confidence interval for the mean brain volume in this new population?

* [1092, 1108]

* [1031, 1169]

* [1080, 1120]

* [1077,1123]

Answer: </br>

```{r}
x_bar <- 1100
s <- 30
n <- 9
alpha <- 0.05
ts <- qt(1 - alpha / 2, n - 1) # 2.306004
round(x_bar + c(-1, 1) * ts * s / sqrt(n))
```

```{r}
# 1077 1123
```

Question 2
----------
A diet pill is given to 9 subjects over six weeks. The average difference in weight (follow up - baseline) is -2 pounds. What would the standard deviation of the difference in weight have to be for the upper endpoint of the 95% T confidence interval to touch 0?

* 0.30

* 1.50

* 2.60

* 2.10

Answer: </br>
```{r}
x_bar <- -2
n <- 9
alpha <- 0.05
ts <- qt(1 - alpha / 2, n - 1) # 2.306004
s <- -x_bar*sqrt(n) / ts
```

```{r}
# 2.601903
```

Question 3
----------
In an effort to improve running performance, 5 runners were either given a protein supplement or placebo. Then, after a suitable washout period, they were given the opposite treatment. Their mile times were recorded under both the treatment and placebo, yielding 10 measurements with 2 per subject. The researchers intend to use a T test and interval to investigate the treatment. Should they use a paired or independent group? T test and interval?

* Independent groups, since all subjects were seen under both systems

* You could use either

* It's necessary to use both

* A paired interval

Answer: </br>
Independent tests: not related participants </br>
Paired tests: related participants, same group uses 2 different tests </br>
A paired interval. </br>

Question 4
----------
In a study of emergency room waiting times, investigators consider a new and the standard triage systems. To test the systems, administrators selected 20 nights and randomly assigned the new triage system to be used on 10 nights and the standard system on the remaining 10 nights. They calculated the nightly median waiting time (MWT) to see a physician. The average MWT for the new system was 3 hours with a variance of 0.60 while the average MWT for the old system was 5 hours with a variance of 0.68. Consider the 95% confidence interval estimate for the differences of the mean MWT associated with the new system. Assume a constant variance. What is the interval? Subtract in this order (New System - Old System).

* [1.25, 2.75]

* [-2,70, -1.29]

* [-2.75, -1.25]

* [1.29, 2.70]

Answer: </br>
```{r}
n_x <- 10 
n_y <- 10
x_bar <- 5 # old_system
y_bar <- 3 # new_system
var_x <- 0.6
var_y <- 0.68
alpha <- 0.05
sp_2 <- ((n_x - 1)*var_x + (n_y - 1)*var_y) / (n_x + n_y - 2)
sp <- sqrt(sp_2)
ts <- qt(1 - (alpha/2), n_x + n_y - 2)
round((y_bar - x_bar) + c(-1, 1) * ts * sp * (sqrt(1/n_x + 1/n_y)), 2) 
```

```{r}
# -2.75 -1.25
```

Question 5
----------
Suppose that you create a 95% T confidence interval. You then create a 90% interval using the same data. What can be said about the 90% interval with respect to the 95% interval?

* It is impossible to tell.

* The interval will be the same width, but shifted.

* The interval will be wider

* The interval will be narrower.

Answer: </br>
The interval will be narrower.

Question 6
----------
To further test the hospital triage system, administrators selected 200 nights and randomly assigned a new triage system to be used on 100 nights and a standard system on the remaining 100 nights. They calculated the nightly median waiting time (MWT) to see a physician. The average MWT for the new system was 4 hours with a standard deviation of 0.5 hours while the average MWT for the old system was 6 hours with a standard deviation of 2 hours. Consider the hypothesis of a decrease in the mean MWT associated with the new treatment.

What does the 95% independent group confidence interval with unequal variances suggest vis a vis this hypothesis? (Because there's so many observations per group, just use the Z quantile instead of the T.)

* When subtracting (old - new) the interval is entirely above zero. The new system does not appear to be effective.

* When subtracting (old - new) the interval contains 0. The new system appears to be effective.

* When subtracting (old - new) the interval contains 0. There is not evidence suggesting that the new system is effective.

* When subtracting (old - new) the interval is entirely above zero. The new system appears to be effective.

Answer: </br>
When subtracting (old - new) the interval is entirely above zero. The new system appears to be effective.

Explanation: 

```{r}
n_x <- 100
n_y <- 100
x_bar <- 6
y_bar <- 4
s_x <- 2
s_y <- 0.5
alpha <- 0.05
sp_2 <- ((n_x - 1)*s_x^2 + (n_y - 1)*s_y^2) / (n_x + n_y - 2)
sp <- sqrt(sp_2)
ts <- qt(1 - (alpha/2), n_x + n_y - 2)
round((x_bar - y_bar) + c(-1, 1) * ts * sp * (sqrt(1/n_x + 1/n_y)), 2) 
```

```{r}
# 1.59 2.41
```

Question 7
----------
Suppose that 18 obese subjects were randomized, 9 each, to a new diet pill and a placebo. Subjects’ body mass indices (BMIs) were measured at a baseline and again after having received the treatment or placebo for four weeks. The average difference from follow-up to the baseline (followup - baseline) was −3 kg/m2 for the treated group and 1 kg/m2 for the placebo group. The corresponding standard deviations of the differences was 1.5 kg/m2 for the treatment group and 1.8 kg/m2 for the placebo group. Does the change in BMI over the four week period appear to differ between the treated and placebo groups? Assuming normality of the underlying data and a common population variance, calculate the relevant *90%* t confidence interval. Subtract in the order of (Treated - Placebo) with the smaller (more negative) number first.

* [2.469, 5.531]

* [2.636, 5.364]

* [-5.364, -2.636]

* [-5.531, -2.469]

Answer: </br>

```{r}
n1 <- 9
n2 <- 9
x1 <- -3 # treated
x2 <- 1 # placebo
s1 <- 1.5 # treated
s2 <- 1.8 # placebo
s <- sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2)/(n1 + n2 - 2))
(x1 - x2) + c(-1, 1) * qt(0.95, n1 + n2 - 2) * s * sqrt(1/n1 + 1/n2)
```

```{r}
# -5.364 -2.636
```

# Statistical Inference Quiz 2 (JHU) Coursera

Question 1
----------
What is the variance of the distribution of the average an IID draw of n observations from a population with mean μ and variance σ<sup>2</sup>.

* σ/n

* σ<sup>2</sup> /n 

* σ<sup>2</sup>

* 2σ/n<sup>.5</sup>

Answer: </br>

σ<sup>2</sup> /n 

Question 2
----------
Suppose that diastolic blood pressures (DBPs) for men aged 35-44 are normally distributed with a mean of 80 (mm Hg) and a standard deviation of 10. About what is the probability that a random 35-44 year old has a DBP less than 70?

* 22%

* 8%

* 32%

* 16%

Answer: </br>

```{r}
pnorm(70, mean = 80, sd = 10)
```

```{r}
## [1] 0.1587
```

Question 3
----------
Brain volume for adult women is normally distributed with a mean of about 1,100 cc for women with a standard deviation of 75 cc. What brain volume represents the 95th percentile?

* approximately 1223

* approximately 1247

* approximately 977

* approximately 1175

Answer: </br>
```{r}
qnorm(0.95, mean = 1100, sd = 75)
```

```{r}
## [1] 1223
```
Question 4
----------
Refer to the previous question. Brain volume for adult women is about 1,100 cc for women with a standard deviation of 75 cc. Consider the sample mean of 100 random adult women from this population. What is the 95th percentile of the distribution of that sample mean?

(Hints, look at quantiles from Lecture 2 around 21:30 and Chapter 2 Problem 7).

* approximately 1112 cc

* approximately 1088 cc

* approximately 1115 cc

* approximately 1110 cc

Answer: </br>

```{r}
qnorm(0.95, mean = 1100, sd = 75/sqrt(100))
```

```{r}
## [1] 1112
```

Question 5
----------
You flip a fair coin 5 times, about what's the probability of getting 4 or 5 heads?

* 12%

* 6%

* 19%

* 3%

Answer: </br>

```{r}
pbinom(3, size = 5, prob = 0.5, lower.tail = FALSE)
```

```{r}
## [1] 0.1875
```

Explanation: I don't find answers like use this random R function too useful especially for interviews where you have to understand the stats, so I wrote it out by hand. 

![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/6_%20Statistical_Inference/data/quizImages/quiz2coinFlipsExplanation.jpg)

Question 6
----------
The respiratory disturbance index (RDI), a measure of sleep disturbance, for a specific population has a mean of 15 (sleep events per hour) and a standard deviation of 10. They are not normally distributed. Give your best estimate of the probability that a sample mean RDI of 100 people is between 14 and 16 events per hour?

* 34%

* 95%

* 68%

* 47.5%

Answer: </br>
The standard error of the mean is 10/100^.5 = 1. Thus between 14 and 16 is with one standard deviation of the mean of the distribution of the sample mean. Thus it should be about 68%.

```{r}
pnorm(16, mean = 15, sd = 1) - pnorm(14, mean = 15, sd = 1)
```

```{r}
## [1] 0.6827
```

Question 7
----------
Consider a standard uniform density. The mean for this density is .5 and the variance is 1 / 12. You sample 1,000 observations from this distribution and take the sample mean, what value would you expect it to be near?

* 0.75

* 0.25

* 0.5

* 0.10

Answer: </br>
Via the LLN it should be near .5.

Question 8
----------
The number of people showing up at a bus stop is assumed to be

Poisson with a mean of 5 people per hour. You watch the bus

stop for 3 hours. About what's the probability of viewing 10 or fewer people?

* 0.12

* 0.06

* 0.08

* 0.03

Answer: </br>

```{r}
ppois(10, lambda = 15)
```

```{r}
## [1] 0.1185
```

# Statistical Inference Quiz 1 (JHU) Coursera

Question 1
----------
Consider influenza epidemics for two parent heterosexual families. Suppose that the probability is 17% that at least one of the parents has contracted the disease. The probability that the father has contracted influenza is 12% while the probability that both the mother and father have contracted the disease is 6%. What is the probability that the mother has contracted influenza?

(Hints look at lecture 2 around 5:30 and chapter 4 problem 4).

* 17%

* 6%

* 11% (answer)

* 5%

Explanation: </br>

For any two events the probability that at least one occurs is the sum of their probabilities minus their intersection. 

![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/6_%20Statistical_Inference/data/unionIntersection.png)
</br>
unions: (denoted by ∪) of a collection of sets is the set of all elements in the collection
</br> A = {1, 3, 5, 7} and B = {1, 2, 4, 6} then A ∪ B = {1, 2, 3, 4, 5, 6, 7}
</br>
intersection: (denoted by ∩) intersection A ∩ B of two sets A and B is the set that contains all elements of A that also belong to B (or equivalently, all elements of B that also belong to A), but no other elements
<br>
A = Mother
<br>
B = Father
<br>
P(A∪B) = 17%
<br>
P(B) = 12%
<br>
P(A∩B) = 6%
<br>
Since we know P(A∪B)=P(A)+P(B)−P(A∩B) we get
<br>
17%=P(A)+12%−6%.
<br>
P(A) = 11%

Question 2
----------
A random variable, X is uniform, a box from 0 to 1 of height 1. (So that its density is f(x)=1 for 0≤x≤1.) What is its 75th percentile?

(Hints, look at lecture 2 around 21:30 and Chapter 5 Problem 5. Also, look up the help function for the qunif command in R.)

* 0.50

* 0.10

* 0.25

* 0.75 (answer)

Explanation: </br>

```{r}
qunif(p=0.75, min = 0, max = 1)
```

```{r}
## [1] 0.75
```

Question 3
----------
You are playing a game with a friend where you flip a coin and if it comes up heads you give her X dollars and if it comes up tails she gives you Y dollars. The probability that the coin is heads is p (some number between 0 and 1.) What has to be true about X and Y to make so that both of your expected total earnings is 0. The game would then be called “fair”.

(Hints, look at Lecture 4 from 0 to 6:50 and Chapter 5 Problem 6. Also, for further reading on fair games and gambling, start with the Dutch Book problem ).

* X = Y

* p = X/Y

* p/(1−p) = X/Y

* p/(1−p) = Y/X (answer)

Explanation: </br>

Your expected earnings is −pX+(1−p)Y=0 Then it must be the case that p1−p=YX Or that the ratio of the payouts has to equal the odds. So consider, for example, if p1−p=2. The game is 2 to 1 against you, p=2/3; she is twice as likely to win as you. Then she will have to pay out twice as much if you win to make the game fair.

p*X = Y*(1-p), therefore p/(1-p)=Y/X.

Question 4
----------
A density that looks like a normal density (but may or may not be exactly normal) is exactly symmetric about zero. (Symmetric means if you flip it around zero it looks the same.) What is its median?

(Hints, look at quantiles from Lecture 2 around 21:30 and Chapter 2 Problem 7).

* We can't conclude anything about the median.

* The median must be different from the mean.

* The median must be 1.

* The median must be 0.

Explanation: </br>

The density is symmetric at zero so the median must be must be 0 (50% of the probability is below 0 and 50% is above 0, when the density is symmetric at zero). 

Question 5
----------
Consider the following PMF shown below in R

```{r}
x <- 1:4
p <- x/sum(x)
temp <- rbind(x, p)
rownames(temp) <- c("X", "Prob")
temp
```

```{r}
## [,1] [,2] [,3] [,4]
## X 1.0 2.0 3.0 4.0
## Prob 0.1 0.2 0.3 0.4
```

What is the mean?

(Hint, watch Lecture 4 on expectations of PMFs.)

* 2

* 1

* 3 (answer)

* 4

```{r}
fi <- temp["X",]
xi <- temp["Prob",] 
  
product <- fi*xi
sum(product)
```

Question 6
----------
A web site (www.medicine.ox.ac.uk/bandolier/band64/b64-7.html) for home pregnancy tests cites the following: “When the subjects using the test were women who collected and tested their own samples, the overall sensitivity was 75%. Specificity was also low, in the range 52% to 75%.” Assume the lower value for the specificity. Suppose a subject has a positive test and that 30% of women taking pregnancy tests are actually pregnant. What number is closest to the probability of pregnancy given the positive test?

(Hints, watch Lecture 3 at around 7 minutes for a similar example. Also, there's a lot of Bayes' rule problems and descriptions out there, for example here's one for HIV testing. Note, discussions of Bayes' rule can get pretty heady. So if it's new to you, stick to basic treatments of the problem. Also see Chapter 3 Question 5.)

* 30%

* 40% (answer)

* 20%

* 10%

Explanation: </br>
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/6_%20Statistical_Inference/data/pregnant.png)

# Machine Learning Week 6 Quiz 1 (Advice for Applying Machine Learning) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/AdviceQ1.png)

Answer | Explanation
--- | ---
High Bias | This learning curve shows high error on both the training and test sets, so the algorithm is suffering from high bias.

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/AdviceQ2.png)

Useful Info: Since the hypothesis performs <b> well </b> (has low error) on the training set, it is suffering from high variance (overfitting)

True/False | Answer | Explanation
--- | --- | ---
False | Try evaluating the hypothesis on a cross validation set rather than the test set. | A cross validation set is useful for choosing the optimal non-model parameters like the regularization parameter λ, but the train / test split is sufficient for debugging problems with the algorithm itself.
False | Try decreasing the regularization parameter λ. | The gap in errors between training and test suggests a high variance problem in which the algorithm has overfit the training set. Decreasing the regularization parameter will increase the overfitting, not decrease it.
True | Try using a smaller set of features. | The gap in errors between training and test suggests a high variance problem in which the algorithm has overfit the training set. Reducing the feature set will ameliorate the overfitting and help with the variance problem
True | Try increasing the regularization parameter λ. |  The gap in errors between training and test suggests a high variance problem in which the algorithm has overfit the training set. Increasing the regularization parameter will reduce overfitting and help with the variance problem.
True | Get more training examples | The gap in errors between training and test suggests a high variance problem in which the algorithm has overfit the training set. Adding more training data will increase the complexity of the training set and help with the variance problem.

Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/AdviceQ3.png)

Useful Info: Since the hypothesis performs <b> poorly </b> on the training set, it is suffering from high bias (underfitting)

True/False | Answer | Explanation
--- | --- | ---
True | Try adding polynomial features. | The poor performance on both the training and test sets suggests a high bias problem. Adding more complex features will increase the complexity of the hypothesis, thereby improving the fit to both the train and test data.
False | Try increasing the regularization parameter λ. | The poor performance on both the training and test sets suggests a high bias problem. Increasing the regularization parameter will allow the hypothesis to fit the data worse, decreasing both training and test set performance.
False | Try using a smaller set of features. | The poor performance on both the training and test sets suggests a high bias problem. Using fewer features will decrease the complexity of the hypothesis and will make the bias problem worse
True | Try to obtain and use additional features. |  The poor performance on both the training and test sets suggests a high bias problem. Using additional features will increase the complexity of the hypothesis, thereby improving the fit to both the train and test data.

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/AdviceQ4.png)

True/False | Answer | Explanation
--- | --- | ---
False | Suppose you are training a regularized linear regression model.The recommended way to choose what value of regularization parameter λ to use is to choose the value of λ which gives the lowest <b>training set</b> error. |  You should not use training error to choose the regularization parameter, as you can always improve training error by using less regularization (a smaller value of ). But too small of a value will not generalize well onthe test set.
True | The performance of a learning algorithm on the training set will typically be better than its performance on the test set. | The learning algorithm finds parameters to minimize training set error, so the performance should be better on the training set than the test set.
True | Suppose you are training a regularized linear regression model. The recommended way to choose what value of regularization parameter λ to use is to choose the value of λ which gives the lowest <b>cross validation</b> error | The cross validation lets us find the "just right" setting of the regularization parameter given the fixed model parameters learned from the training set. 
False | Suppose you are training a regularized linear regression model. The recommended way to choose what value of regularization parameter λ to use is to choose the value of λ which gives the lowest <b>test set</b> error. |  You should not use the test set to choose the regularization parameter, as you will then have an artificially low value for test error and it will not give a good estimate of generalization error.
True | A typical split of a dataset into training, validation and test sets might be 60% training set, 20% validation set, and 20% test set. | This is a good split of the data, as it dedicates the bulk of the data to finding model parameters in training while leaving enough data for cross validation and estimating generalization error.

Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/AdviceQ5.png)

True/False | Answer | Explanation
--- | --- | ---
True | If a learning algorithm is suffering from high bias, only adding more training examples may <b>not</b> improve the test error significantly. |  If a learning algorithm is suffering from high bias, only adding more training examples may not improve the test error significantly.
True | A model with more parameters is more prone to overfitting and typically has higher variance. | More model parameters increases the model's complexity, so it can more tightly fit data in training, increasing the chances of overfitting.
True | When debugging learning algorithms, it is useful to plot a learning curve to understand if there is a high bias or high variance problem. | The shape of a learning curve is a good indicator of bias or variance problems with your learning algorithm.
False | If a neural network has much lower training error than test error, then adding more layers will help bring the test error down because we can fit the test set better. | With lower training than test error, the model has high variance. Adding more layers will increase model complexity, making the variance problem worse.

# Machine Learning Week 6 Quiz 2 (Machine Learning System Design) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/MachineLearningSystemDesignQ1.png)

Statistic | Solution
--- | ---
Accuracy | (85 + 10) / (1000) = .095
Precision | (85) / (85 + 890) = 0.087
Recall | There are 85 true positives and 15 false negatives, so recall is 85 / (85 + 15) = 0.85.
F<sub>1</sub> Score | (2 * (0.087 * 0.85)) / (0.087 + 0.85) = 0.16

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/MachineLearningSystemDesignQ2.png)

True/False | Answer | Explanation
--- | --- | ---
False | We train a learning algorithm with a small number of parameters (that is thus unlikely to overfit). | 
True | We train a learning algorithm with a large number of parameters (that is able tolearn/represent fairly complex functions). | You should use a "low bias" algorithm with many parameters, as it will be able to make use of the large dataset provided. If the model has too few parameters, it will underfit the large training set.
True | The features x contain sufficient information to predict y accurately. (For example, one way to verify this is if a human expert on the domain can confidently predict y when given only x). | It is important that the features contain sufficient information, as otherwise no amount of data can solve a learning problem in which the features do not contain enough information to make an accurate prediction.
False | We train a model that does not use regularization. |  Even with a very large dataset, some regularization is still likely to help the algorithm's performance, so you should use cross-validation to select the appropriate regularization parameter.
False | The classes are not too skewed. | The problem of skewed classes is unrelated to training with large datasets.
True | Our learning algorithm is able to represent fairly complex functions (for example, if we train a neural network or other model with a large number of parameters). | You should use a complex, "low bias" algorithm, as it will be able to make use of the large dataset provided. If the model is too simple, it will underfit the large training set.
False | When we are willing to include high order polynomial features of x | As we saw with neural networks, polynomial features can still be insufficient to capture the complexity of the data, especially if the features are very high-dimensional. Instead, you should use a complex model with many parameters to fit to the large training set.
True | A human expert on the application domain can confidently predict y when given only the features x (or more generally we have some way to be confident that x contains sufficient information to predict y accuratly) | This is a nice project commencement briefing. 

Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/MachineLearningSystemDesignQ3.png)

True/False | Answer | Explanation
--- | --- | ---
True | The classifier is likely to now have lower recall. | Increasing the threshold means more y = 0 predictions. This increase  will decrease the number of true positives and increase the number of false negatives, so recall will decrease.
False | The classifier is likely to have unchanged precision and recall, but lower accuracy. | By making more y = 0 predictions, we decrease true and false positives and increase true and false negatives. Thus, precision and recall will certainly change. We cannot say whether accuracy will increase or decrease.
False | The classifier is likely to have unchanged precision and recall, but thus the same F<sub>1</sub> score. | By making more y = 0 predictions, we decrease true and false positives and increase true and false negatives. Thus, precision and recall will certainly change. We cannot say whether the F<sub>1</sub> score will increase or decrease.
False | The classifier is likely to now have lower precision. |  Increasing the threshold means more y = 0 predictions. This will decrease both true and false positives, so precision will increase, not decrease.

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/MachineLearningSystemDesignQ4.png)

True/False | Answer | Explanation
--- | --- | ---
True | If you always predict non-spam (output y = 0 ), your classifier will have a recall of 0%. |  Since every prediction is y = 0, there will be no true positives, so recall is 0%.
False | If you always predict spam (output y = 1), your classifier will have a recall of 0% and precision of 99%. | Every prediction is y = 1, so recall is 100% and precision is only 1%.
True | If you always predict non-spam (output y = 0), your classifier will have 99% accuracy on the training set, and it will likely perform similarly on the cross validation set. | The classifier achieves 99% accuracy on the training set because of how skewed the classes are. We can expect that the cross-validation set will be skewed in the same fashion, so the classifier will have approximately the same accuracy. 
True | If you always predict spam (output y = 1), your classifier will have a recall of 100% and precision of 1%. |  Since every prediction is y = 1, there are no false negatives, so recall is 100%. Furthermore, the precision will be the fraction of examples with are positive, which is 1%.
False | A good classifier should have both a high precision and high recall on the cross validation set. | Different classifiers are made for different reasons with different goals. 
True | If you always predict non-spam (output y=0), your classifier will have an accuracy of 99%. | none needed. 

Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/MachineLearningSystemDesignQ5p1.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/MachineLearningSystemDesignQ5p2.png)

True/False | Answer | Explanation
--- | --- | ---
True | Using a <b>very large</b> training set makes it unlikely for model to overfit the training data | A sufficiently large training set will not be overfit, as the model cannot overfit some of the examples without doing poorly on the others.
False | It is a good idea to spend a lot of time collecting a <b>large</b> amount of data before building your first version of a learning algorithm. | You cannot know whether a huge dataset will be important until you have built a first version and find that the algorithm has high variance.
False | After training a logistic regression classifier, you <b>must</b> use 0.5 as your threshold for predicting whether an example is positive or negative. |  You can and should adjust the threshold in logistic regression using cross validation data.
False | If your model is underfitting the training set, then obtaining more data is likely to help. |  If the model is underfitting the training data, it has not captured the information in the examples you already have. Adding further examples will not help any more.
True | The "error analysis" process of manually examining the examples which your algorithm got wrong can help suggest what are good steps to take (e.g., developing new features) to improve your algorithm's performance. | This process of error analysis is crucial in developing high performance learning systems, as the space of possible improvements to your system is very large, and it gives you direction about what to work on next.
True | On skewed datasets (e.g., when there are more positive examples than negative examples), accuracy is not a good measure of performance and you should instead use F1 score based on the precision and recall. | This is a wonderful interview question. 
True | The error analysis process of manually examining the examples which your algorithm got wrong can help suggest what are good steps to take (e.g. developing new features) to improve your algorithm's performance | none needed

# Machine Learning Week 7 Quiz 1 (Support Vector Machines) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ1p1.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ1p2.png)

Answer | Explanation
--- | ---
It would be reasonable to try decreasing C. It would also be reasonable to try increasing  σ<sup>2</sup>  | The figure shows a decision boundary that is overfit to the training set, so we'd like to increase the bias / lower the variance of the SVM. We can do so by either decreasing the parameter C or increasing σ<sup>2</sup>

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ2p1.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ2p2.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ2p3.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ2p4.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ2p5.png)


Answer

![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ2p2Solution.png)

Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ3.png)

True/False | Answer | Explanation
--- | --- | ---
False | For every example with y<sup>(i)</sup>=1, we have that Θ<sup>T</sup>x<sup>(i)</sup>>= 0  | cost<sub>0</sub>(Θ<sup>T</sup>x<sup>(i)</sup>) is still non-zero for inputs between 0 and 1, so being greater than or equal to 0 is insufficient. 
False | For every example with y<sup>(i)</sup>=0, we have that Θ<sup>T</sup>x<sup>(i)</sup><= 0 | cost<sub>0</sub>(Θ<sup>T</sup>x<sup>(i)</sup>) is still non-zero for inputs between -1 and 0, so being less than or equal to 0 is insufficient. 
True | For every example with y<sup>(i)</sup>=0, we have that Θ<sup>T</sup>x<sup>(i)</sup><= -1  |  For examples with y<sup>(i)</sup>=0, only the cost<sub>0</sub>(Θ<sup>T</sup>x<sup>(i)</sup>) term is present. As you can see in the graph, this will be zero for all inputs greater than or equal to -1. 
True | For every example with y<sup>(i)</sup>=1, we have that Θ<sup>T</sup>x<sup>(i)</sup>>= 1  | For examples with y<sup>(i)</sup>=1,only the cost<sub>1</sub>(Θ<sup>T</sup>x<sup>(i)</sup>) term is present. As you can see in the graph, this will be zero for all inputs greater than or equal to 1. 

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ4.png)

True/False | Answer | Explanation
--- | --- | ---
False | Use an SVM with a linear kernel, without introducing new features.| An SVM with only the linear kernel is comparable to logistic regression, so it will likely underfit the data as well. 
True | Use an SVM with a Gaussian Kernel| By using a Gaussian kernel, your model will have greater complexity and you can avoid underfitting the data. 
False | Increase the regularization parameter λ | You are already underfitting the data and increasing the regularization parameter only makes underfitting stronger. 
True | Create / add new polynomial features | When you add more features, you increase the variance of your model, reducing your chances of underfitting. 
False | Use a different optimization method since gradient descent to train logisitic regression might result in a local minimum | The logistic regression cost function is convex, so gradient descent will always find the global minimum. 
True | Try using a neural network with a large number of hidden units | A neural network with many hidden units is a more complex (higher variance) model than logistic regression, so it is less likely to underfit the data. 

Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/SupportVectorMachinesQ5.png)

True/False | Answer | Explanation
--- | --- | ---
False | If the data are linear separable, an SVM using a linear kernel will return the same parameters Θ regardless of the chosen value of C (i.e., the resulting value of Θ does not depend on C)  | A linearly separable dataset can usually be separated by many different lines. Varying the parameter C will cause the SVM's decision boundary to vary among these possibilities. For example, for a very large value of C, it might learn larger values of Θ in order to increase the margin on certain examples. 
True | The maximum value of the Gaussian kernel (i.e., sim(x,l<sup>(1)</sup>)) is 1. | When x = l<sup>(1)</sup>, the Gaussian kernel has value exp(0) = 1, and it is less than 1 otherwise. 
True | Suppose you had 2D input examples (i.e. x<sup>(i)</sup> ∈ R<sup>2</sup>). The decision boundary of the SVM (with the linear kernel) is a straight line. | The SVM without any kernel (i.e., the linear kernel) predicts output based only Θ<sup>T</sup>x, so it gives a linear/straight-line decision boundary, just as logisitic regression does. 
False | If you are training multi-class SVMs with the one-vs-all method, it is not possible to use a kernel | Each SVM you train in the one-vs-all method is a standard SVM, so you are free to use a kernel. 
False | Suppose you are using SVMs to do multi-class classification and would likely to use the one-vs-all approach. If you have K different classes, you will train K-1 different SVMs. | The one-vs-all method requires that we have a separate classifier for every class, so you will train K different SVMs.
True | It is important to perform feature normalization before using the Gaussian kernel. | The similarity measure used by the Gaussian kernel expects that the data lie in approximately the same range.  

# Machine Learning Week 8 Quiz 1 (Unsupervised Learning) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/unsupervisedLearningQ1.png)

True or False | Statement | Explanation 
--- | --- | ---
False | Given historical weather records, predict if tomorrow's weather will be sunny or rainy | K-means cannot make classification predictions as it does not label its inputs.
True | Given a set of news articles from many different websites, find out what topics are the main topics covered | You can use K-means to cluster, and each cluster will correspond to a different market segment. 
True | From the user usage patterns on a website, figure out what different groups of users exists. | You can use K-means to cluster users with each cluster corresponding to a different market segment. 
False | Given many emails, you want to determine if they are Spam or Non-Spam emails.  | K-means cannot make classification predictions as it does not label its inputs
True | Given a database of information about your users, automatically group them into different market segments. | You can use K-means to cluster the database entries, and each cluster will correspond to a different market segment.
True | Given sales data from a large number of products in a supermarket, figure out which products tend to form coherent groups (say are frequently purchased together) and thus should be put on the same shelf. | Market Segmentation.
False | Given sales data from a large number of products in a supermarket, estimate future sales for each of these products. | Such a prediction is a regression problem, and K-means does not use labels on the data, so it cannot perform regression.

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/unsupervisedLearningQ2.png)

Answer | Explanation 
--- | --- 
c<sup>(i)</sup> = 1 | x<sup>(i)</sup> is closest to μ<sub>1</sub>, so c<sup>(i)</sup> = 1

Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/unsupervisedLearningQ3.png)

True or False | Statement | Explanation 
--- | --- | ---
False | Randomly initialize the cluster centroids | Done earlier
False | Test on the cross-validation set | Any sort of testing is outside the scope of K-means algorithm itself
True | Move the cluster centroids, where the centroids, μ<sub>k</sub> are updated | The cluster update is the second step of the K-means loop 
True | The cluster assignment step, where the parameters c<sup>(i)</sup> are updated | This is the correst first step of the Kmeans loop

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/unsupervisedLearningQ4.png)

Answer | Explanation 
--- | --- 
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/ClusteringDistortionFunction.png" alt="Distortion Cost FUnction"> | This is the distortion cost function which we seek to minimize

Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/unsupervisedLearningQ5.png)

True or False | Statement | Explanation 
--- | --- | ---
False | Once an example has been assigned to a particular centroid, it will never be reassigned to another centroid | Not sure yet
True | A good way to initialize K-means is to select K (distinct) examples from the training set and set the cluster centroids equal to these selected examples. | This is the recommended method of initialization.
True | On every iteration of K-means, the cost funtion J(c<sup>(1)</sup>, ..., c<sup>(m)</sup>,  μ<sub>1</sub>, ...,  μ<sub>k</sub> (the distortion function) should either stay the same or decrease; in particular, it should not increase | True  
False | K-Means will always give the same results regardless of the initialization of the centroids. | K-means is sensitive to different initializations, which is why you should run it multiple times from different random initializations
True | For some datasets, the "right" or "correct" value of K (the number of clusters) can be ambiguous, and hard even for a human expert looking carefully at the data to decide. | Look at an elbow curve for an example. It can often be ambiguous.
True | If we are worried about K-means getting stuck in bad local optima, one way to ameliorate (reduce) this problem is if we try using multiple random initializations. | None needed

# Machine Learning Week 8 Quiz 2 (Principle Component Analysis) Stanford Coursera

Question 1
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq1part1.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq1part2.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq1part3.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq1part4.png)
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq1part5.png)

Answer | Explanation 
--- | --- 
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAQ1answer1.png" alt="Maximal variance image"> | The maximal variance is along the y = x line, so this option is correct.
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAQ1answer2.png" alt="Maximal variance image"> | The maximal variance is along the y = x line, so the negative vector along that line is correct for the first principal component

Question 2
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq2.png)

Answer | Explanation 
--- | --- 
Choose k to be the smallest value so that at least 99% of the variance is retained | This maintains the structure of the data while maximally reducing its dimension.


Question 3
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq3.png)

Answer | Explanation 
--- | --- 
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq3Answer.png" alt="Maximal variance image"> | It is just a formula. 

Question 4
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq4.png)

Answer | Explanation 
--- | --- 
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq4Answer1.png" alt="PCA Q4 Answer 1"> | If you do not perform mean normalization, PCA will rotate the data in a possibly undesired way.
<img src="https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq4Answer2.png" alt="PCA Q4 Answer 2"> | None needed

Question 5
----------
![](https://github.com/mGalarnyk/datasciencecoursera/blob/master/Stanford_Machine_Learning/Week1/data/PCAq5.png)

True or False | Statement | Explanation 
--- | --- | ---
False | Data visualization: To take 2D data, and find a different way of plotting it in 2D (using k=2) | None needed
False | As a replacement for (or alternative to) linear regression: For most learning applications, PCA and linear regression give substantially similar results | PCA is not linear regression. They have different goals (and cost functions), so they give different results.
True | Data compression: Reduce the dimension of your input data x<sup>(i)</sup>, which will be used in a supervised learning algorithm (i.e., use PCA so that your supervised learning algorithm runs faster) | If your learning algorithm is too slow because the input dimension is too high, then using PCA to speed it up is a reasonable choice.
True | Data compression: Reduce the dimension of your data, so that it takes up less memory/disk space. | If memory or disk space is limited, PCA allows you to save space in exchange for losing a little of the data's information. This can be a reasonable tradeoff.
