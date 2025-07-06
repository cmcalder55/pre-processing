
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def onehot_encoder_pipeline(categorical_cols, numeric_cols):
    """Applies a column transformer to categorical and numeric features.
    Categorical features are one-hot encoded, while numeric features are standardized.

    The `remainder` parameter is set to 'passthrough' to keep any other columns
    that are not explicitly listed in `categorical_cols` or `numeric_cols`.

    Args:
        categorical_cols (list): List of categorical column names.
        numeric_cols (list): List of numeric column names.

    Returns:
        ColumnTransformer: A ColumnTransformer that applies OneHotEncoder to categorical columns
        and StandardScaler to numeric columns.
    """
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )

# Custom PyTorch Dataset
class TorchData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural Network Model
class TorchNetModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(TorchNetModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # Keep shape [batch_size, 1]

def train_model(model, train_loader, val_loader, val_dataset, num_epochs=10, lr=0.001):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == y_batch).sum().item()

        val_accuracy = correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    return model
        