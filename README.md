# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## THEORY


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load data, create word/tag mappings, and group sentences.

### STEP 2: 
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.



### STEP 3: 
Define dataset and DataLoader for batching.



### STEP 4: 
Build a bidirectional LSTM model for sequence tagging.



### STEP 5: 
Train the model over multiple epochs, tracking loss



### STEP 6: 
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.





## PROGRAM

### Name:GEDIPUDI DARSHANI

### Register Number:212223230062

```python
# Model definition
class BiLSTMTagger(nn.Module):
  def __init__(self,vocab_size,tarset_size,embedding_dim=50,hidden_dim=100):
    super(BiLSTMTagger,self).__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.dropout=nn.Dropout(0.1)
    self.lstm=(nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True))
    self.fc=nn.Linear(hidden_dim*2,tarset_size)
  def forward(self,x):
    x=self.embedding(x)
    x=self.dropout(x)
    x,_=self.lstm(x)
    return self.fc(x)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)   # FIXED
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss += loss.item()  # FIXED

        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

    return train_losses, val_losses



```

### OUTPUT

## Loss Vs Epoch Plot

<img width="620" height="502" alt="image" src="https://github.com/user-attachments/assets/d4269021-e0f1-4033-90c3-31a0f7fc7acf" />

### Sample Text Prediction
<img width="355" height="350" alt="image" src="https://github.com/user-attachments/assets/8abf39b6-cfec-4e1f-8862-ef7d26bb4c7b" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
