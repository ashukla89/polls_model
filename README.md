# polls-model
This is a stab at a model that would have calculated daily polling predictions for each of four parties in Dataland, in one of five scenarios. Sadly, thanks to lots of red herrings in the model design, it's never really worked.

Look at `polling_model.py` if you want to know where I was going with this. The structure of the app should be apparent from there.

## Where time fell short
Most obviously, this app doesn't quite serve to generate predictions, although it's nearly there.

What is much less near is a procedure for Monte Carlo simulations to develop probability distributions around the predicted outcomes.