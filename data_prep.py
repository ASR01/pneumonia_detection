import torch
import numpy as np
from PIL import Image
import os

img_size = 640
torch.manual_seed(0)
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu" )

class LungsImageDataset(torch.utils.data.Dataset):
	
    def __init__(self, image, target):

        self.image = image 
        self.target = target 

    def __len__(self):
        return len(self.target) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = self.image[idx]
        target = self.target[idx]
        
    
        return target, image



def get_data_in_dl(path):
    
    print("Building Dataset")
    x , y = [], []

    for folder in os.listdir(path):
        x , y = [], []
        p1 = os.path.join(path, folder)
        print(folder)
        for folder2 in os.listdir( os.path.join(path, folder) ):
            p2 = os.path.join(p1, folder2)
            #print(p2)
            for file in os.listdir(p2):
                #print(file)
                
                img = Image.open( os.path.join(p2, file) )
                #w, h = img.size
                #print(w,h)
                img = img.convert("L").resize((img_size,img_size)) #B&W and resize to 640x640
                img = np.array(img).astype(np.float32)
                
                img = torch.from_numpy(img).reshape(1,img_size,img_size)
                #img = img.type(torch.DoubleTensor)
                x.append(img)
                y.append( 1 if folder2 == "PNEUMONIA" else 0 )

        x = torch.stack(x)
        y = np.array(y)
        y = torch.from_numpy(y).float()
        y = y.unsqueeze(1)
        dl = LungsImageDataset(x,y)
        torch.save(dl, './dataloader/' + folder + '_images.pkl')
    
    return dl


def main():
    
    dl = get_data_in_dl('./data/')
    
if __name__ == '__main__':
    main()
