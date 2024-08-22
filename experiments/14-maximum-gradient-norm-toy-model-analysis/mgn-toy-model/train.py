import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

from data_model import MeanEstimator


def train_dp_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    max_grad_norm,
    epsilon,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Wrap model and optimizer with Opacus for DP
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=1e-5,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        normalize_clipping=True,
    )

    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    clipped_proportions = []

    # Training and validation loop
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0

        # Train
        for batch in train_loader:
            optimizer.zero_grad()

            # We have to extract the batch data this way,
            # because we don't have any targets
            batch_data = batch[0]

            # Output is the predicted mean
            output = model(batch_data)

            # Compute the loss as MSE between predicted mean and actual data
            loss = criterion(output, batch_data)
            loss.backward()

            # Calculate the proportion of clipped gradients
            total_gradients = 0
            clipped_gradients = 0
            for param in model.parameters():
                if param.grad_sample is not None:
                    total_gradients += param.grad_sample.numel()
                    clipped_gradients += (
                        (param.grad_sample.abs() > max_grad_norm).sum().item()
                    )

            clipped_proportion = (
                clipped_gradients / total_gradients if total_gradients > 0 else 0
            )
            clipped_proportions.append(clipped_proportion)

            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                val_data = batch[0]
                val_output = model(val_data)
                val_loss = criterion(
                    val_output, val_data
                )
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    # Unwrap the vanilla model from Opacus
    vanilla_model = model.to_standard_module()

    # Record the norm of the predicted mean
    predicted_means_norm_train = torch.norm(vanilla_model.mean).item()
    predicted_means_norm_val = torch.norm(vanilla_model.mean).item()

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'clipped_proportions': clipped_proportions,
        'predicted_means_norm_train': predicted_means_norm_train,
        'predicted_means_norm_val': predicted_means_norm_val,
    }
