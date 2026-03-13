import math

def detect_diagnostics(history):
    diagnostics = []

    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    # Overfitting
    if train_losses[-1] < train_losses[0] and val_losses[-1] > val_losses[0]:
        diagnostics.append({
            "type": "Overfitting",
            "message": "Training loss decreases while validation loss increases. Model is memorizing training data."
        })

    # Underfitting
    if train_losses[-1] > 0.5 and val_losses[-1] > 0.5:
        diagnostics.append({
            "type": "Underfitting",
            "message": "Both training and validation loss remain high. Model is too simple."
        })

    # Unstable training
    for loss in train_losses + val_losses:
        if math.isnan(loss) or loss > 1e6:
            diagnostics.append({
                "type": "Unstable training",
                "message": "Loss exploded or became NaN. Learning rate may be too high."
            })
            break

    return diagnostics
