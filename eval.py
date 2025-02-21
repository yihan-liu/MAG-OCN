# eval.py

import numpy as np
import matplotlib.pyplot as plt

metrics = np.load('./metrics/metrics.npz')

train_losses = metrics['train_losses']
test_losses = metrics['test_losses']
train_r2s = metrics['train_r2s']
test_r2s = metrics['test_r2s']

epochs = range(1, len(train_losses) + 1)

# Plot the loss curves.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve')
plt.legend()

# Plot the R² curves.
plt.subplot(1, 2, 2)
plt.plot(epochs, train_r2s, label='Train R²')
plt.plot(epochs, test_r2s, label='Test R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('R² Curve')
plt.legend()

plt.tight_layout()
plt.show()