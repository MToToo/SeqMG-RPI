import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix, f1_score, average_precision_score
from torch.optim.lr_scheduler import StepLR


# Function to calculate evaluation index
def calculate_metrics(y_true, y_pred):
    # Check the number of categories in y_true
    unique_labels = set(y_true)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class ({unique_labels}) in y_true. "
              f"evaluation metrics: MCC, AUC and AUPR cannot be calculated.")
        tn, fp, fn, tp = None, None, None, None  # Confusion matrix cannot be defined
        specificity = None
        mcc = None
        auc = None
        aupr = None
    else:
        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        aupr = average_precision_score(y_true, y_pred)

    # Always calculate the following metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)  # Avoid errors when the denominator is 0
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Return result dictionary
    return {
        'Acc': accuracy,
        'Sens': sensitivity,
        'Spec': specificity,  # If it cannot be evaluated, it will be None
        'Pre': precision,
        'MCC': mcc,  # If it cannot be evaluated, it will be None
        'AUC': auc,  # If it cannot be evaluated, it will be None
        'F1': f1,
        'AUPR': aupr  # If it cannot be evaluated, it will be None
    }


# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cpu', patience=20):
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Learning rate scheduler, multiplying the learning rate by 0.1 every 5 epochs
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0  # Best validation set accuracy
    epochs_no_improve = 0  # Record how many consecutive epochs there is no improvement

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        all_labels = []
        all_preds = []

        for rna_features, rna_kmer, rna_svd, protein_batch, labels, _ in train_loader:
            rna_features, rna_kmer, rna_svd, protein_batch, labels = rna_features.to(device), rna_kmer.to(device), rna_svd.to(device), protein_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(rna_features, rna_kmer, rna_svd, protein_batch)
            loss = loss_fn(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * rna_features.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend((outputs.squeeze() > 0.5).cpu().tolist())

        scheduler.step()  # Update learning rate

        # Calculate the evaluation index of the training set
        train_metrics = calculate_metrics(all_labels, all_preds)
        # Calculate the loss and evaluation index of the validation set and the predicted probability value
        val_loss, val_metrics, val_labels, val_probs, val_ids = evaluate_model(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch + 1}/{num_epochs} ")
        print(f"Loss: {epoch_loss / len(train_loader.dataset)}, T Acc: {train_metrics['Acc']}")
        print(f"V Loss: {val_loss}")
        print(f"V Acc: {val_metrics['Acc']}, Sens: {val_metrics['Sens']}, ")
        print(f"Spec: {val_metrics['Spec']}, Pre: {val_metrics['Pre']}, ")
        print(f"MCC: {val_metrics['MCC']}, AUC: {val_metrics['AUC']}, ")
        print(f"F1: {val_metrics['F1']}, AUPR: {val_metrics['AUPR']}")
        # print(f"Validation Predictions: {val_probs}")  # Output the predicted probability value of the validation set

        # Save the best model weights
        if val_metrics['AUC'] > best_val_acc:
            best_val_acc = val_metrics['AUC']
            torch.save(model.state_dict(), '../best_model.pth')
            epochs_no_improve = 0  # Reset the early stop counter
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("Training complete")
    # Load the best model weights
    model.load_state_dict(torch.load('../best_model.pth'))
    return model


# Function to evaluate the model
def evaluate_model(model, dataloader, loss_fn, device='cpu'):
    model.eval()
    total_loss = 0.0
    all_labels = []  # Store all true labels
    all_probs = []  # Store all predicted interaction probability values
    all_preds = []  # Store all predicted labels (True, False)
    all_ids = []   # Store all id information

    with torch.no_grad():
        for rna_features, rna_kmer, rna_svd, protein_batch, labels, ids in dataloader:
            rna_features, rna_kmer, rna_svd, protein_batch, labels = rna_features.to(device), rna_kmer.to(device), rna_svd.to(device), protein_batch.to(device), labels.to(device)
            outputs = model(rna_features, rna_kmer, rna_svd, protein_batch)
            probs = outputs.squeeze()  # Obtaining predicted interaction probability values
            loss = loss_fn(outputs.squeeze(), labels)
            total_loss += loss.item() * rna_features.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend((outputs.squeeze() > 0.5).cpu().tolist())
            all_ids.extend(ids)

    metrics = calculate_metrics(all_labels, all_preds)
    return total_loss / len(dataloader.dataset), metrics, all_labels, all_probs, all_ids
