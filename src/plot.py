import matplotlib.pyplot as plt
import pickle

with open('loss_history.pkl', 'rb') as f:
    loss_history = pickle.load(f)

train_loss = loss_history['train_loss'][::2]
train_psnr = loss_history['train_psnr'][::2]
val_psnr = loss_history['val_psnr'][::2]

epochs = range(1,len(train_loss) + 1)

plt.subplot(1,3,1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,3,2)
plt.plot(epochs, train_psnr, label='Train PSNR', color='orange')
plt.title('Training PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

plt.subplot(1,3,3)
plt.plot(epochs, val_psnr, label='Test PSNR', color='red')
plt.title('Test PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

plt.tight_layout()
plt.show()

print(len(train_loss))
