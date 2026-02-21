import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import re
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def get_items(metadata, idx):
    subject_ID = metadata.loc[idx, 'subject_id']
    studyt_ID = metadata.loc[idx, 'study_id']
    image_ID = metadata.loc[idx, 'dicom_id']
    view_position = metadata.loc[idx, 'ViewPosition']
    img_height = metadata.loc[idx, 'Rows']
    img_width = metadata.loc[idx, 'Columns']

    return subject_ID, studyt_ID, image_ID, view_position, img_height, img_width

def create_img_path(parent_path, subject_ID, study_ID, image_ID):

    patient_parent_folder = (str(subject_ID))[:2]
    path_to_img = parent_path + "/p" + patient_parent_folder + "/p" + str(subject_ID) + "/s" + str(study_ID) + "/" + image_ID + ".jpg"

    return path_to_img

def create_report_path(parent_path, subject_ID, study_ID):

    patient_parent_folder = (str(subject_ID))[:2]
    path_to_img = parent_path + "/p" + patient_parent_folder + "/p" + str(subject_ID) + "/s" + str(study_ID) + ".txt"

    return path_to_img


class XRay_Dataset(Dataset):
    def __init__(self, dataset_path, metadata_path, transform=None, metadata_subset=None):
        self.dataset_path = dataset_path
        self.metadata_path = metadata_path
        
        # If metadata_subset is provided, use it; otherwise, load the entire metadata
        if metadata_subset is not None:
            self.metadata = metadata_subset
        else:
            self.metadata = pd.read_excel(metadata_path)

        self.grouped = self.metadata.groupby('study_id').agg(lambda x: list(x))
        self.len = len(self.grouped)

        # Define transform if not passed
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize image to match the ViT input size
                transforms.ToTensor(),  # Convert image to tensor
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = self.grouped.iloc[idx]
        lateral = None
        lateral_path = None
        pa = None
        pa_path = None
        for i in range(len(item['dicom_id'])):
            image_ID = item['dicom_id'][i]
            study_ID = item.name
            subject_ID = item['subject_id'][i]
            view_position = item['ViewPosition'][i]
            image_path = create_img_path(self.dataset_path, subject_ID, study_ID, image_ID)
            study_report_path = create_report_path(self.dataset_path, subject_ID, study_ID)

            # Open as PIL, then apply Torchvision transforms
            img = Image.open(image_path).convert("RGB")
            image = self.transform(img)

            if view_position == "PA":
                pa_image = image
                pa_path = image_path
            elif view_position == "LATERAL":
                lateral = image
                lateral_path = image_path

        with open(study_report_path, 'r') as file:
            content = file.read()
        
        findings = "FINDINGS: " + content.split("FINDINGS:")[1].split("IMPRESSION:")[0].split("CONCLUSION:")[0].strip()
        findings = re.sub(r'\s+', ' ', findings).strip()

        # Determine whether the next section is IMPRESSION or CONCLUSION
        if "IMPRESSION:" in content:
            impression = "IMPRESSION: " + content.split("IMPRESSION:")[1].strip()
        elif "CONCLUSION:" in content:
            impression = "IMPRESSION: " + content.split("CONCLUSION:")[1].strip()
        else:
            impression = "IMPRESSION: " + "NaN"
        
        """
        disease_info = self.disease_labels.loc[study_ID].to_dict() if study_ID in self.disease_labels.index else {}
        disease_info = (
            {
                **self.disease_labels.loc[study_ID].drop(['study_id', 'subject_id'], errors='ignore').dropna().to_dict()
            }
            if study_ID in self.disease_labels.index
            else {}
        )
        """


        # Create study dictionary
        study_dict = {
            "Patient_ID": subject_ID,
            "Study_ID": study_ID,
            "Lateral": lateral,
            "Lateral_path": lateral_path,
            "PA": pa_image,
            "PA_path": pa_path,
            "Report": content,
            "Findings": findings,
            "Impression": impression,
            #"Diseases": disease_info,
        }
        return study_dict
    
if __name__ == "__main__":
    cleaned_dataset_dir = os.environ.get("DATASET_DIR", "data/cleaned_dataset")
    output_excel_file = os.environ.get("METADATA_FILE", "data/cleaned_dataset_metadata.xlsx")
    
    dataset = XRay_Dataset(dataset_path=cleaned_dataset_dir, metadata_path=output_excel_file)
    print(len(dataset))

    study_dict, pa_image, findings, impression = dataset[0]
    print(findings)
