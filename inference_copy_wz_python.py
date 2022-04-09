import os
from pathlib import Path
from fastai.learner import load_learner
import numpy as np


class CombinedLoss:
    "Dice and Focal combined"
    def __init__(self, axis=1, smooth=1., alpha=1.,gamma=3):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis,gamma=gamma)
        self.dice_loss =  DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)



def label_func(x:Path):
    return f"{combined_mask2}/{x.stem}.png"

def main():
    combined_img2 = Path('C:/Users/thomas/Desktop/irdis/AISpark_Challenge_IRDIS-main/AIHub/Training/train_buildings_data2')
    combined_mask2 = Path('C:/Users/thomas/Desktop/irdis/AISpark_Challenge_IRDIS-main/AIHub/Training/train_buildings_labeling3')
    model_1_path = Path().resolve() / "predict_footprint_fine_tuned"
    model_2_path = Path().resolve() / "predict_damage_unfrozen"
    model_wz_path = Path().resolve() / "predict_wz_footprint_fine_tuned"

    predict_footprint = load_learner(model_1_path)
    predict_damage = load_learner(model_2_path)
    predict_footprint_wz = load_learner(model_wz_path)
    aftertif = 'inference_data/after/10300100C540A500_-90.021_29.89.tif'
    beforetif = 'inference_data/before/105001001E0A3300_-90.021_29.89.tif'
    print(os.path.isfile(aftertif))
    print(os.path.isfile(beforetif))
    result = predict_footprint.predict(item=beforetif)[0]
    np.save("inference_wz_result",result)
    
if __name__ == "__main__":
    main()
    