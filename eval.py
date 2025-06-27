import torch
def evaluate(model, data_loader, loss_fn, accuracy_fn,device):
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name":model.__class__.__name__,
            "model_loss": loss.item(),
            "model_accuracy": acc}