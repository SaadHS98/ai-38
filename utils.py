import torch
import torchvision
from data import CarvanaDataset
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F

NUM_FOLDS = 5

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    fold_index=None,

):


    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=val_transform,
    )

    val_start_index = len(val_ds) - 200
    val_subset = Subset(val_ds, range(val_start_index, len(val_ds)))

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    # train_ds = CarvanaDataset(
    #     image_dir=train_dir,
    #     mask_dir=train_maskdir,
    #     transform=train_transform,
    # )
    #
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=True,
    # )
    #
    # val_ds = CarvanaDataset(
    #     image_dir=val_dir,
    #     mask_dir=val_maskdir,
    #     transform=val_transform,
    # )
    #
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )
    if fold_index is not None:
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        train_indices, val_indices = list(kf.split(train_ds))[fold_index]
        train_ds = Subset(train_ds, train_indices)
        val_ds = Subset(val_ds, val_indices)

    return train_loader, val_loader

smooth = 1.
dropout_rate = 0.5

def mean_iou(y_true, y_pred):
    prec = []
    for t in torch.arange(0.5, 1.0, 0.05):
        y_pred_ = (y_pred > t).int()
        intersection = torch.sum(y_true * y_pred_)
        union = torch.sum(y_true) + torch.sum(y_pred_) - intersection
        score = intersection / (union + smooth)
        prec.append(score)
    return torch.mean(torch.stack(prec), dim=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * F.binary_cross_entropy(y_pred, y_true) - dice_coef(y_true, y_pred)
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()).item() / ((preds + y).sum().item() + 1e-8)

    accuracy = num_correct / num_pixels * 100
    mean_dice = dice_score / len(loader)

    # Calculate mean_iou and bce_dice_loss
    mean_iou_score = mean_iou(y, preds)
    bce_dice = bce_dice_loss(y, preds)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Mean IoU Score: {mean_iou_score:.4f}")
    print(f"BCE + Dice Loss: {bce_dice:.4f}")
    model.train()


# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device).unsqueeze(1)
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2 * (preds * y).sum()) / (
#                 (preds + y).sum() + 1e-8
#             )
#
#     print(
#         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
#     )
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()