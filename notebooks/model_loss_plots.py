# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mer.utils.utils import plot_history

history_path = "../history/crnn_3_fold_3.npy"
history_path_2 = "../history/crnn_3_fold_3_sep.npy"

# Plot
with open(history_path, "rb") as f:
  [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)


# Plot
with open(history_path_2, "rb") as f:
  [epochs_loss_s, epochs_val_loss_s] = np.load(f, allow_pickle=True)



# %%


e_all_loss = []

id = 0
time_val = []
for epoch in epochs_loss:
  for step in epoch:
    e_all_loss.append(step.numpy())
    id += 1
  # time_val.append(id)

# plt.figure(facecolor='white')
plt.plot(np.arange(0, len(e_all_loss)*7, 7), e_all_loss, label = "Training loss")
plt.plot(np.arange(0, len(e_all_loss)*7, 7), epochs_val_loss, label = "Validation loss")

# plt.plot(np.arange(1,len(e_loss)+ 1), e_loss, label = "train loss")
# plt.plot(np.arange(1,len(epochs_val_loss)+ 1), epochs_val_loss, label = "val loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()

plt.show()


# %%

mean_loss = []
mean_val_loss = []

mean_loss_sep = []
mean_val_loss_sep = []

for i in range(10):
  history_path = f"../history/crnn_3_fold_{i}.npy"
  history_path_2 = f"../history/crnn_3_fold_{i}_sep.npy"
  with open(history_path, "rb") as f:
    [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)

  e_all_loss = []
  for epoch in epochs_loss:
    for step in epoch:
      e_all_loss.append(step.numpy())
  e_all_val_loss = []
  for val in epochs_val_loss:
    e_all_val_loss.append(val.numpy())
  mean_loss.append(e_all_loss)
  mean_val_loss.append(e_all_val_loss)
  
  with open(history_path_2, "rb") as f:
    [epochs_loss_s, epochs_val_loss_s] = np.load(f, allow_pickle=True)

  e_all_loss_s = []
  for epoch in epochs_loss_s:
    for step in epoch:
      e_all_loss_s.append(step.numpy())
  e_all_val_loss_s = []
  for val in epochs_val_loss_s:
    e_all_val_loss_s.append(val.numpy())
  mean_loss_sep.append(e_all_loss_s)
  mean_val_loss_sep.append(e_all_val_loss_s)

mean_loss = tf.reduce_mean(mean_loss, axis=0)
mean_val_loss = tf.reduce_mean(mean_val_loss, axis=0)

mean_loss_sep = tf.reduce_mean(mean_loss_sep, axis=0)
mean_val_loss_sep = tf.reduce_mean(mean_val_loss_sep, axis=0)

# %%

plt.figure()
fig, axes = plt.subplots(2,1, figsize=(4, 5))

# plt.figure(facecolor='white')
axes[0].plot(np.arange(0, len(mean_loss)*7, 7), mean_loss, label = "Training loss")
axes[0].plot(np.arange(0, len(mean_val_loss)*7, 7), mean_val_loss, label = "Validation loss")
# axes[0].xlabel("Step")
# axes[0].ylabel("Loss")
axes[0].legend()

axes[1].plot(np.arange(0, len(mean_loss_sep)*7, 7), mean_loss_sep, label = "Training loss")
axes[1].plot(np.arange(0, len(mean_val_loss_sep)*7, 7), mean_val_loss_sep, label = "Validation loss")
# axes[1].xlabel("Step")
# axes[1].ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()

print(mean_val_loss[-1])
print(mean_val_loss_sep[-1])

# %%

# Plot separately

plt.figure(figsize=(6,3))

# plt.figure(facecolor='white')
plt.plot(np.arange(0, len(mean_loss)*7, 7), mean_loss, label = "Training loss")
plt.plot(np.arange(0, len(mean_val_loss)*7, 7), mean_val_loss, label = "Validation loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(6,3))
plt.plot(np.arange(0, len(mean_loss_sep)*7, 7), mean_loss_sep, label = "Training loss")
plt.plot(np.arange(0, len(mean_val_loss_sep)*7, 7), mean_val_loss_sep, label = "Validation loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()



# %%
