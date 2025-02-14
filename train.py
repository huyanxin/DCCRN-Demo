import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dc_crn import DCCRN
from loss import SISNRLoss
from dataset import VCTKDEMANDDataset
from tqdm import tqdm

# Configuration parameters
batch_size = 1
num_epochs = 200
learning_rate = 0.001
checkpoint_dir = 'checkpoints'
data_dir = '/home/qianjingrui0827/qjr_projects/gtcrn/VCTK-DEMAND'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

os.makedirs(checkpoint_dir, exist_ok=True)


def collate_fn(batch):
    """Pads batch of variable length Tensors."""
    # Find the longest sequence in the batch
    max_length = max([x[0].shape[1] for x in batch])

    # Pad sequences to the max length
    padded_noisy = torch.zeros(len(batch), 1, max_length)
    padded_clean = torch.zeros(len(batch), 1, max_length)

    for i, (noisy, clean) in enumerate(batch):
        length = noisy.shape[1]
        padded_noisy[i, 0, :length] = noisy
        padded_clean[i, 0, :length] = clean

    return padded_noisy, padded_clean


# Instantiate the dataset and DataLoader
train_dataset = VCTKDEMANDDataset(root_dir=data_dir)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Instantiate the model and loss function
model = DCCRN().to(device).train()
loss_func = SISNRLoss().to(device)

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with early stopping based on loss threshold
loss_threshold = 0.5  # Desired loss threshold

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_waveform, clean_waveform in tqdm(train_loader):
        noisy_waveform = noisy_waveform.to(device)
        clean_waveform = clean_waveform.to(device)

        optimizer.zero_grad()

        # Forward pass
        _, outputs = model(noisy_waveform)

        outputs = outputs.unsqueeze(1)

        # 计算所需的填充长度
        padding_length = clean_waveform.size(-1) - outputs.size(-1)

        # 填充 outputs 到目标形状
        if padding_length > 0:
            outputs = torch.nn.functional.pad(outputs, (0, padding_length), mode="constant", value=0)
        loss = loss_func(outputs, clean_waveform)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    if avg_loss <= loss_threshold:
        print(f"Stopping training as loss is below {loss_threshold}")
        break 

torch.save({'model': model.state_dict()}, os.path.join(
    checkpoint_dir, f'dccrn_trained_on_vctk_jrqian{epoch+1}.pt'))


print("Training complete and model saved")
