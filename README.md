# linear-regression-ml-model-with-generating-cases
Machine Learning Model predicts value Y using X1 and X2.

What This Program Does
This script is a predictive analytics tool that simulates real-world behavior using a neural network model.

In simple terms:
It generates synthetic data for two input variables: X1 and X2.

Then it trains a neural network to learn the relationship between X1, X2, and a calculated Y.

Once trained, it can predict Y when you input new values for X1 and X2.

It also gives you confidence intervals to tell you how certain the model is about its predictions.

It saves predictions to an Excel file, and optionally launches a web UI where you can play with sliders to see predictions live.

What the Program Predicts
It predicts the value of Y based on inputs X1 and X2.

Example:
Suppose you input:

X1 = 5.0

X2 = 7.0

The model will calculate something like:

Predicted Y = 30.5

Lower 95% CI = 27.0

Upper 95% CI = 34.0

So, it says:

“I think Y is about 30.5, but I’m 95% confident it lies between 27.0 and 34.0.”

What Is Confidence Interval (CI)?
Confidence Interval (CI) tells you how confident the model is in its prediction.

In this program:
We simulate 100 predictions with slight randomness (like dropout layers).

Then we take the:

Mean → Predicted value

2.5th percentile → Lower bound

97.5th percentile → Upper bound

This gives you a 95% CI (confidence interval).

In simple words:
“If I made this prediction 100 times, 95 of those results would fall between the lower and upper bounds.”

How the Prediction Logic Works
The formula to simulate Y is:

ini
Copy
Edit
Y = 3.5 * X1 + 2.1 * X2 + noise
But your model doesn't know the formula — it learns the pattern from the data.

Then it can make predictions on new input.

Does It Learn Over Time?
Yes. Every time the script runs:

It loads the saved model (if it exists),

And trains it further on the new data.

This means the model gradually improves as it sees more data.

