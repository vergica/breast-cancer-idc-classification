import os
import glob
import pandas as pd
import torch
import multiprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image


class IDCDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row["label"], dtype=torch.long)
    
def get_optimal_workers():
    try:
        cpu_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_cores = 2
    return min(8, max(0, cpu_cores - 2))

def get_dataloaders(data_dir, batch_size=64, test_size=0.2):

    # 1. 扫描文件提取数据
    all_paths = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    data_list = []
    for path in all_paths:
        filename = os.path.basename(path)
        data_list.append({
            "path": path,
            "patient_id": filename.split("_")[0],
            "label": 1 if "class1" in filename else 0
        })

    df = pd.DataFrame(data_list)
    print(f"在这个路径下一共找到了 {len(df)} 张图片")
    
    # 2. 按患者id划分训练集和测试集
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["patient_id"]))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df["patient_id"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"数据划分：训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

    # 3. 数据增强与预处理
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),          
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 实例化Dataset并封装为DataLoader 

    train_dataset = IDCDataset(train_df, transform=train_transform)
    val_dataset = IDCDataset(val_df, transform=test_transform)
    test_dataset = IDCDataset(test_df, transform=test_transform)
    ## 加入WeightedRandomSampler
    class_counts = train_df["label"].value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = train_df["label"].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    auto_workers = get_optimal_workers()
    has_gpu = torch.cuda.is_available()
    print(f"num_workers={auto_workers}, pin_memory={has_gpu}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=auto_workers,
        pin_memory=has_gpu,
        persistent_workers=(auto_workers > 0)
    )

     
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=auto_workers,
        pin_memory=has_gpu,
        persistent_workers=(auto_workers > 0)
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=auto_workers,
        pin_memory=has_gpu,
        persistent_workers=(auto_workers > 0)
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
