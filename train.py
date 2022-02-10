#utils
from cgi import print_directory
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

#torch
from torch.optim import Adam
from torch import nn
import torch

#files
from model import CNN_classifier_S
from data_prep import LungsImageDataset





def validate(test_loader, model, criterion, device):

    with torch.no_grad():
    
        val_loss = 0
        model.eval()
        acc_sum = 0
        for i, (y, x) in enumerate(test_loader):                 


            x, y = x.to(device), y.to(device)
            y = y.type(torch.int64).flatten() # Cross entropy needs integers
             
            output = model(x)
            
            #Get the predictions appl;ying the softmax
            pred = []
      
            for out in output:
                softmax = torch.exp(out).cpu()
                prob = list(softmax.numpy())
                pred.append(np.argmax(prob, axis=0))
            pred = np.array(pred)

            acc = accuracy_score(pred, y)
            
            acc_sum += acc
            loss = criterion(output, y)

            val_loss += loss.cpu().item()

        return val_loss / len(test_loader), acc_sum/len(test_loader)




def train(epochs, train_loader, test_loader, model, criterion, optim, device, scheduler):

    print("---TRAINING---")

    best_val = float('INF')
    best_acc =1.0
    model.train()
    train_losses = []
    val_losses = []
    accuracy = []

    for epoch in range( epochs ):

        
        loss_iter = 0
        

        for i, (y, x) in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            
            x, y = x.to(device), y.to(device)
            y = y.type(torch.int64).flatten() # Cross entropy needs integers
            
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            optim.step()
            loss_iter += loss.cpu().item()
        
        train_losses.append(loss_iter) 
        scheduler.step() 
        val_loss, val_acc = validate(test_loader, model, criterion, device)
        print("Epoch: {} \t Train Loss: {} \t Validation Loss: {} \t Validation Acc: {}". format( epoch, loss_iter/len(train_loader), val_loss, val_acc*100) )
        val_losses.append(val_loss)
        accuracy.append(val_acc)

        if epoch+1 % 2:

            
            if val_loss < best_val:
                print("Saving Validation Model")
                best_val = val_loss
                torch.save(model, "models/best_val_model_c.pt")
            
            if val_acc < best_acc:
                print("Saving Accuracy Model")
                best_acc = val_acc
                torch.save(model, "models/best_acc_model_c.pt")
            
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.savefig("figures/Losses.png")


    

    
def main():

	#Device
	device = torch.device("cuda" if  torch.cuda.is_available() else "cpu" )

	#Dataloaders

	train_ds = torch.load('./dataloader/train_images.pkl')
	test_ds = torch.load('./dataloader/test_images.pkl')	
	# train_ds = torch.load('./test_dataloader/train_images.pkl')
	# test_ds = torch.load('./test_dataloader/test_images.pkl')

	train_loader = torch.utils.data.DataLoader(train_ds, shuffle = True , batch_size = 16 , drop_last = True)
	test_loader = torch.utils.data.DataLoader(test_ds, shuffle = True , batch_size = 16 , drop_last = True)

	#HyperParams
	epochs = 30
	lr = 0.001
	dropout = 0.2

	#Models
	model = CNN_classifier_S(dropout)
	#print( summary( model, (1,512,512) ) )

	criterion = nn.CrossEntropyLoss()
	optim = Adam(model.parameters(), lr )
	scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.50)

	#Device
	model = model.to(device)
	criterion = criterion.to(device)

	#TRAINING

	train(epochs, train_loader, test_loader, model, criterion, optim, device, scheduler)

if __name__ == '__main__':
    main()

