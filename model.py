from torch import nn
from torchsummary import summary



class CNN_classifier_M(nn.Module):

    def __init__(self, dropout = 0.2):
        super(CNN_classifier_M, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout, inplace = True),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout, inplace = True),
            nn.ReLU(inplace = True),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout, inplace = True),
            nn.ReLU(inplace = True),
        )

        self.linear = nn.Sequential(
            nn.Linear(128*81*81, 128),
            nn.Dropout(dropout, inplace = True),
            # nn.Linear(512, 128),
            # nn.Dropout(dropout, inplace = True),
            nn.Linear(128, 16),
            nn.Dropout(dropout, inplace = True),
            nn.Linear(16, 2),
        )





    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x  = self.linear(x)
        return x


class CNN_classifier_S(nn.Module):
	
    def __init__(self, dropout = 0.2):
        super(CNN_classifier_S, self).__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.Dropout(dropout, inplace = True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, kernel_size = 5, stride = 2, padding = 1),
            nn.Dropout(dropout, inplace = True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 1),
            nn.Dropout(dropout, inplace = True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )

        self.linear = nn.Sequential(
            nn.Linear(64*79*79, 128),
            nn.Dropout(dropout, inplace = True),

            nn.Linear(128, 16),
            nn.Dropout(dropout, inplace = True),

            nn.Linear(16, 2),

        )

    def forward(self, x):

        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x  = self.linear(x)
        return x



# Check Layers
def main(): 
    model = CNN_classifier_S()
    print( summary( model, (1,640,640) ) )
    print(model)

if __name__ == '__main__':
    main()