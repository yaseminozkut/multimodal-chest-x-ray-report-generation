import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from xray_dataset import XRay_Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

accumulation_steps = 4  # Define it globally

def get_data_loaders(metadata_file, dataset_dir, batch_size, drop_last=False):
    # Load metadata
    metadata_df = pd.read_excel(metadata_file)

    # Split studies
    unique_studies = metadata_df['study_id'].unique()
    train_studies, temp_studies = train_test_split(unique_studies, test_size=0.2, random_state=42)
    val_studies, test_studies = train_test_split(temp_studies, test_size=0.5, random_state=42)

    # Create subsets
    train_metadata = metadata_df[metadata_df['study_id'].isin(train_studies)]
    val_metadata = metadata_df[metadata_df['study_id'].isin(val_studies)]
    test_metadata = metadata_df[metadata_df['study_id'].isin(test_studies)]

    # Create datasets
    train_dataset = XRay_Dataset(dataset_dir, None, transform=None, metadata_subset=train_metadata)
    val_dataset = XRay_Dataset(dataset_dir, None, transform=None, metadata_subset=val_metadata)
    test_dataset = XRay_Dataset(dataset_dir, None, transform=None, metadata_subset=test_metadata)

# Data loaders (without DistributedSampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    """
    # Debugging
    if accelerator.is_main_process:
        effective_batch_size = batch_size * accelerator.num_processes * accumulation_steps
        steps_per_epoch = len(train_loader) // accumulation_steps
        print(f"Train Dataset Size: {len(train_dataset)}")
        print(f"Validation Dataset Size: {len(val_dataset)}")
        print(f"Test Dataset Size: {len(test_dataset)}")
        print(f"Batch Size Per GPU: {batch_size}")
        print(f"Accumulation Steps: {accumulation_steps}")
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Effective Batch Size: {effective_batch_size}")
        print(f"Steps per Epoch (from DataLoader): {len(train_loader)}")
        print(f"True Steps per Epoch (after Accumulation): {steps_per_epoch}")
        """


    return train_loader, val_loader, test_loader
