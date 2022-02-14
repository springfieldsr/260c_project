from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mean = torch.tensor([0.4914, 0.4822, 0.4465])
std = torch.tensor([0.2009, 0.2009, 0.2009])
resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

cifar10_train_data = ShuffledDataset('CIFAR10', './data', 0.1, train=True, transform=resnet_transform)
cifar10_test_data = ShuffledDataset('CIFAR10', './data', 0, train=False, transform=resnet_transform)

test_loader = DataLoader(cifar10_test_data, Config.BATCH_SIZE, shuffle=False)

# Import resnet18 pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)

train(model, Config.EPOCH, cifar10_train_data, test_loader, device)
accuracy = eval(model, test_loader, device)
print(accuracy)