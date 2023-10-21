import torch.optim as optim
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


# Train the model
def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()

    # Setup train loass and train accuracy values
    train_loss, train_acc = 0, 0
    
    for batch, (X,y) in enumerate(dataloader):
      # Send data to target Device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate the loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item()

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and acculate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metric to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def valid_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  # Put model in eval mode
  model.eval()

  # Setup test loss and test accuracy values
  valid_loss, valid_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X,y) in enumerate(dataloader):
      # Send the data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      valid_pred_logits = model(X)

      # 2. Calculate and accumulate loss
      loss = loss_fn(valid_pred_logits,y)
      valid_loss += loss.item()

      # 3. Calculate and accumulate the acuracy
      valid_pred_labels = valid_pred_logits.argmax(dim=1)
      valid_acc += ((valid_pred_labels== y).sum().item()/ len(valid_pred_labels))

  # Adjust metrics to get average loss and accuracy 
  valid_loss = valid_loss / len(dataloader)
  valid_acc = valid_acc / len(dataloader)
  return valid_loss, valid_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device : torch.device
          ) -> Dict[str, List]:
  
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "valid_loss": [],
      "valid_acc": []
  }


  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
    valid_loss, valid_acc = valid_step(model=model,
                                     dataloader=valid_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)
    # Print out what's happening
    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"valid_loss: {valid_loss:.4f} | "
          f"valid_acc: {valid_acc:.4f}"
      )

    # Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["valid_loss"].append(valid_loss)
    results["valid_acc"].append(valid_acc)

  # Return the filled results at the end of the epochs
  return results
