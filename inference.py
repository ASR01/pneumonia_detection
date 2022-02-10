import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path

model = torch.load('./models/best_acc_model_g.pt', map_location=torch.device('cpu'))

trf = transforms.Compose([
     transforms.Resize((640, 640)),
     transforms.ToTensor()
     ])


def predict(image):

	model.eval()
	image = Image.open(image)

	image = image.convert("L") #BW

	input = trf(image)
	input = input.view(1, 1, 640,640)

	with torch.no_grad():
		output = model(input)
		print(output)
		softmax = torch.exp(output).cpu()
		prob = list(softmax.numpy())
		print(prob)
		prediction = np.argmax(prob)
		print(prediction)

		if (prediction == 0):
			pr_text = 'Clean'
		if (prediction == 1):
			pr_text = 'Pneumonia'
	return pr_text



