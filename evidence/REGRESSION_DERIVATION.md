# Linear regression derivation notes (Member 2)

These notes match `src/regression.py`. The implementation does not call a library regressor.

## Hypothesis

For a row with scaled features \(x_1,\dots,x_n\):

\[
h_\theta(x) = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n = \mathbf{x}_b^\top \theta
\]

\(\theta_0\) is the intercept. The bias column of ones is added after scaling so that the intercept is not distorted by feature magnitudes.

## Loss

Batch mean squared error with the conventional \(1/2\) factor:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
\]

## Gradient

In matrix form, with design matrix \(X\) that already includes the bias column:

\[
\nabla J(\theta) = \frac{1}{m} X^\top (X\theta - y)
\]

## Update

\[
\theta \leftarrow \theta - \alpha \nabla J(\theta)
\]

`LEARNING_RATE` (\(\alpha\)) and `N_ITERATIONS` live in `src/config.py` so they can be changed in the live check. The loss history is plotted in `artifacts/regression_loss.png`. A successful run should show a decreasing curve on the training split.

## Scaling and split

1. Shuffle with a fixed seed (`RANDOM_SEED = 42`).
2. Hold out a test fraction (`TEST_SIZE = 0.2`, at least one row).
3. Compute feature mean and standard deviation on **training rows only**.
4. Apply those statistics to the test rows.
5. Report MAE, RMSE, and \(R^2\) on the test predictions.

If a feature has zero training standard deviation, the code replaces that standard deviation with 1 so scaling does not divide by zero.

## Metrics

- MAE: mean absolute residual in kilograms
- RMSE: square root of mean squared residual in kilograms
- \(R^2 = 1 - SS_{res}/SS_{tot}\). If the test target has zero variance, \(R^2\) is reported as 0 rather than NaN.
