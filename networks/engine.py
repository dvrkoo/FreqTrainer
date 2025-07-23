import torch
from tqdm import tqdm
from utils import preprocess_batch  # Import from our new utils


def train_one_epoch(model, data_loader, loss_fun, optimizer, device, args, epoch_num):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(
        data_loader, desc=f"Epoch {epoch_num} [Training]", disable=not args.pbar
    )
    for batch in pbar:
        optimizer.zero_grad()

        # Assuming standard single-dataset loading
        images = batch[data_loader.dataset.key].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        images = preprocess_batch(images, args)

        if args.nclasses == 2:
            labels[labels > 0] = 1  # Binarize labels

        out = model(images)
        loss = loss_fun(out, labels)

        loss.backward()
        optimizer.step()

        acc = (out.argmax(dim=-1) == labels).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()

        pbar.set_postfix(loss=loss.item(), acc=f"{acc.item():.4f}")

    return total_loss / len(data_loader), total_acc / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, loss_fun, device, args, description="Validation"):
    """Runs evaluation on a dataset."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(data_loader, desc=description, disable=not args.pbar)
    for batch in pbar:
        images = batch[data_loader.dataset.key].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        images = preprocess_batch(images, args)

        if args.nclasses == 2:
            labels[labels > 0] = 1

        out = model(images)
        loss = loss_fun(out, labels)
        acc = (out.argmax(dim=-1) == labels).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()

        pbar.set_postfix(loss=loss.item(), acc=f"{acc.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)

    print(f"{description} Results -> Accuracy: {avg_acc:.4f}, Loss: {avg_loss:.4f}")
    return avg_acc, avg_loss
