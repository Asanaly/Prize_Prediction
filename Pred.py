# k-blocks

import numpy as np

k = 4
samples_in_one = len(train_data) // k
num_epochs = 80
all_mae_histories = []
all_scores = []

for i in range(k):
  print(f"Doing good at fold # {i}")
  val_data = train_data[i * samples_in_one : (i + 1) * samples_in_one]
  val_targets = train_labels[i * samples_in_one : (i + 1) * samples_in_one]

  partial_train_data = np.concatenate(
      [train_data[: i * samples_in_one],
      train_data[(i + 1) * samples_in_one:]],
      axis = 0
  )
  partial_train_labels = np.concatenate(   
      [train_labels[: i * samples_in_one],
      train_labels[(i + 1) * samples_in_one:]],
      axis = 0
  )
  model = build_models()
  history = model.fit(partial_train_data , partial_train_labels, epochs = num_epochs, batch_size = 16 , verbose = 0)

  all_mae_histories.append( history.history['mae'] )

  val_mse, val_mae = model.evaluate(val_data , val_targets, verbose = 0)
  
