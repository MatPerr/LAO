import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from lao.embedding.autoencoder import ArcAE, CosineAnnealingAlphaLR
from lao.embedding.graph_dataloader import MultiEpochsDataLoader
from lao.graph.graph_utils import ArcGraph
from lao.graph.search_space import SearchSpace

os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if __name__ == "__main__":
    BO_device = "cpu"
    cnn_device = "mps"
    search_space = SearchSpace()
    ae = ArcAE(search_space=search_space, z_dim=99, ae_type="WAE")
    checkpoint = torch.load("checkpoints/arcae_20250405_103101/arcae_final.pt", map_location=BO_device)
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.bounds = None
    ae.to(BO_device)
    ae.eval()
    with torch.no_grad():
        z = torch.load("nas_models/canonical_20250409_092658/z_vectors/best_z_at_iteration_120.pt")
        z = z.to(BO_device)
        v = ae.decode(z.unsqueeze(0))
        v = v.squeeze(0)
        g = ArcGraph(search_space=search_space, V=v, n_nodes=20)
        g2 = g.to_blueprint(input_shape=(3, 32), enforce_max_preds=True)
        g2.plot(display=True)
        model = g.to_torch(input_shape=(3, 32), num_classes=10, enforce_max_preds=True)
        summary(model, input_size=(3, 32, 32))
        print(model.FLOPs)
        model = model.to(cnn_device)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))]
    )
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = MultiEpochsDataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    max_iterations = 10000
    validation_frequency = max_iterations // 20
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"trained_models/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs", f"model_training_{timestamp}"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = CosineAnnealingAlphaLR(optimizer, T_max=max_iterations, alpha=0.0001)
    best_acc = 0.0
    iteration = 0
    model.train()
    pbar = tqdm(total=max_iterations, desc="Training...")
    while iteration < max_iterations:
        for x, y in train_loader:
            x, y = (x.to(cnn_device), y.to(cnn_device))
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            _, predicted = outputs.max(1)
            acc = predicted.eq(y).sum().item() / y.size(0)
            pbar.set_description(
                f"Iter {iteration + 1} | loss {loss.item():.2e} | acc {100 * acc:.2f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )
            pbar.update(1)
            writer.add_scalar("train/loss", loss.item(), iteration)
            writer.add_scalar("train/accuracy", acc, iteration)
            writer.add_scalar("train/LR", optimizer.param_groups[0]["lr"], iteration)
            if iteration % validation_frequency == 0 or iteration == max_iterations - 1:
                model.eval()
                correct = 0
                total = 0
                val_loss = 0.0
                with torch.no_grad():
                    for val_x, val_y in test_loader:
                        val_x, val_y = (val_x.to(cnn_device), val_y.to(cnn_device))
                        val_outputs = model(val_x)
                        val_loss += criterion(val_outputs, val_y).item()
                        _, val_predicted = val_outputs.max(1)
                        total += val_y.size(0)
                        correct += val_predicted.eq(val_y).sum().item()
                val_acc = correct / total
                avg_val_loss = val_loss / len(test_loader)
                writer.add_scalar("val/loss", avg_val_loss, iteration)
                writer.add_scalar("val/accuracy", val_acc, iteration)
                print(
                    f"\nValidation at iteration {iteration}: Accuracy: {val_acc * 100:.2f}%, Loss: {avg_val_loss:.4f}"
                )
                if val_acc > best_acc:
                    best_acc = val_acc
                    print(f"New best accuracy: {best_acc * 100:.2f}%")
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                model.train()
            iteration += 1
            if iteration >= max_iterations:
                break
    pbar.close()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = (x.to(cnn_device), y.to(cnn_device))
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    final_acc = correct / total
    print(f"\nFinal Test Accuracy: {final_acc * 100:.2f}%")
    print(f"Best Test Accuracy: {best_acc * 100:.2f}%")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "final_accuracy": final_acc,
            "best_accuracy": best_acc,
            "n_params": model.n_params,
            "FLOPs": model.FLOPs,
            "BBGP": model.BBGP if hasattr(model, "BBGP") else None,
        },
        os.path.join(save_dir, "final_model.pth"),
    )
    g2.plot(output_path=os.path.join(save_dir, "architecture.png"))
    print(f"Model saved to {save_dir}")
    writer.close()
