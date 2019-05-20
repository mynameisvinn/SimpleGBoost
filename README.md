# SimpleGBoost
bare bones implementation of gradient boosting

## use
same easy api as scikit's
```python

model = SimpleGBoost()  # instantiate model
model.fit(X_train, y_train)  # fit to data, scikit style
pred = model.predict()  # generate predictions
```