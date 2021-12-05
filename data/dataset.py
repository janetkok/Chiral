from PIL import Image
import pandas as pd
import torch
import numpy as np

def has_transparency(img):
    """ ***************************************************************************************
    *    Title: check for image transparency
    *    Author: Vinyl Da.i'gyu-Kazotetsu
    *    Date: 26th Oct 2019
    *    Availability: https://stackoverflow.com/a/58567453
    *
    *************************************************************************************** """
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False


class MolDataset(torch.utils.data.Dataset):
    """custom multilabel dataset 
    Arguments:
        ds: dataset containing path to images and their respective labels
        split_idx: indices of samples to be used
        transparent2white: creating white background for image with transparency 
        color2grayscale: convert all colour images to grayscale
        aug: perform augmentation
        transforms: basic transformation
        transformsAug: transformation with augmentation
    """
    def __init__(self, ds, split_idx, transparent2white = False, color2grayscale = False,
    aug=False, transforms=None, transformsAug=None):

        self.ds = ds.iloc[split_idx, :]
        self.transparent2white = transparent2white
        self.color2grayscale = color2grayscale
        self.transforms = transforms
        self.transformsAug = transformsAug
        self.aug = aug

    def __getitem__(self, idx):
        d = self.ds.iloc[idx]
        image = Image.open(d.image_path)
        if self.transparent2white:
            image = image.convert("RGBA")
            if has_transparency(image):
                w_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
                w_image.paste(image, (0, 0), image)              # Paste the image on the background. 
                image = w_image
        if self.color2grayscale:
            im_gray = np.array(image.convert('L'))
            im_bin_keep = (im_gray > 140) * im_gray
            image = Image.fromarray(np.uint8(im_bin_keep))
        image = image.convert("RGB")
        l = d[1:].tolist()
        label = torch.tensor(l, dtype=torch.float32)
        if self.aug == False:
            image = self.transforms(image)
        else:
            image = self.transformsAug(image)

        return image, label

    def __len__(self):
        return len(self.ds)

    def get_labels(self):
        dlabel = self.ds.drop(columns=["image_path"])
        tlabel = np.empty((len(dlabel),), dtype=object)
        tlabel[:] = [tuple(i) for i in dlabel.values]
        return tlabel

    def get_label_count(self):
        dlabel = self.ds.drop(columns=["image_path"])
        label_count = dlabel[dlabel == 1].sum(axis=0)
        return label_count.to_numpy()

    def filename(self, idx, basename=False):
        d = self.ds.iloc[idx]
        if basename:
            return d.image_path.split('ral/')[-1]
        else:
            return d.image_path

    def filenames(self):
        return self.ds[['image_path']].to_numpy()


class BinaryDataset(torch.utils.data.Dataset):
    """custom binary dataset 
    Arguments:
        ds: dataset containing path to images and their respective labels
        split_idx: indices of samples to be used
        transparent2white: creating white background for image with transparency 
        color2grayscale: convert all colour images to grayscale
        transforms: basic transformation
    """
    def __init__(self, ds, split_idx, transparent2white = False, color2grayscale = False, transforms=None):
        self.ds = ds.iloc[split_idx, :]
        self.transparent2white = transparent2white
        self.color2grayscale = color2grayscale
        self.transforms = transforms

    def __getitem__(self, idx):
        d = self.ds.iloc[idx]
        image = Image.open(d.image_path)
        if self.transparent2white:
            image = image.convert("RGBA")
            if has_transparency(image):
                w_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
                w_image.paste(image, (0, 0), image)              # Paste the image on the background. 
                image = w_image
        if self.color2grayscale:
            im_gray = np.array(image.convert('L'))
            im_bin_keep = (im_gray > 140) * im_gray
            image = Image.fromarray(np.uint8(im_bin_keep))
        image = image.convert("RGB")

        label = torch.tensor(d.label, dtype=torch.long)
        image = self.transforms(image)

        return image, label

    def filename(self, idx, basename=False):
        d = self.ds.iloc[idx]
        if basename:
            return d.image_path.split('ral/')[-1]
        else:
            return d.image_path
    def filenames(self):
        return self.ds[['image_path']].to_numpy()
    def __len__(self):
        return len(self.ds)