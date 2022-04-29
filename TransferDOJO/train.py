import torch
import tqdm
from ray import tune
import os

from TransferDOJO import evaluation
from TransferDOJO import losses


def train_simclr(
    model,
    optimizer,
    criterion,
    data_loader,
    num_epochs,
    writer,
    save_model_dir,
    temperature=0.1,
    snapshot_dir=None
):
    def info_nce_loss(features):

        labels = torch.cat(
            [torch.arange(data_loader.batch_size) for i in range(2)], dim=0
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / temperature
        return logits, labels

    pbar = tqdm.tqdm(range(num_epochs))
    for epoch_counter in pbar:
        running_loss = 0.0
        for images, _ in data_loader:
            images = torch.cat(images, dim=0)
            images = images.cuda()
            features = model(images)
            logits, labels = info_nce_loss(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        pbar.set_description(f"Loss: {running_loss / len(data_loader):.6f}")
        if writer:
            writer.add_scalar(
                "loss/simclr", running_loss / len(data_loader), epoch_counter
            )
        if epoch_counter % 100 == 0 or epoch_counter == num_epochs - 1:
            custom_models.save_model(
                model, os.path.join(save_model_dir, f"{epoch_counter}.pt")
            )
            print(f"model saved to: {save_model_dir}")

    return model


def train_classification(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    num_epochs,
    writer,
    num_outputs,
    reprd
):
    pbar = tqdm.tqdm(range(num_epochs))
    # CE for classification
    criterion = torch.nn.CrossEntropyLoss()
    # training
    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # put on gpu
            inputs, labels = data["X"].cuda(), data["y"].cuda()
            optimizer.zero_grad()
            # outputs = model(inputs)
            outputs = torch.nn.functional.softmax(model(inputs), dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if reprd and (not epoch % reprd.step_size):
                reprd.store_batch(representation=outputs, ids=ata["index"], epoch=epoch)

        if scheduler:
            scheduler.step()
        # get validation loss #
        with torch.no_grad():
            val_running_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data["X"].cuda(), data["y"].cuda()
                # outputs = model(inputs)
                outputs = torch.nn.functional.softmax(model(inputs), dim=1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        pbar.set_description(f"Loss: {running_loss / len(train_loader):.6f}")
        print(f"running loss: {running_loss}")
        train_results = evaluation.evaluate(
            model, train_loader, train_strategy="train_classification"
        )
        val_results = evaluation.evaluate(
            model, val_loader, train_strategy="train_classification"
        )

        if writer:
            writer.add_scalar("loss/train", running_loss / len(train_loader), epoch)
            writer.add_scalar("accuracy/train", train_results["accuracy"], epoch)
            writer.add_scalar("auc/train", train_results["auc"], epoch)
            writer.add_scalar("loss/val", val_running_loss / len(val_loader), epoch)
            writer.add_scalar("accuracy/val", val_results["accuracy"], epoch)
            writer.add_scalar("auc/val", val_results["auc"], epoch)
    return model
