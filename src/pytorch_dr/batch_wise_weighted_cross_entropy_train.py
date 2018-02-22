from __future__ import print_function, division
import matplotlib
matplotlib.use("TKAgg")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import folder as custom
# import custom2
# import nibabel as nib
import torchvision.models
from torchvision.models.densenet import model_urls
import torchnet as tnt
from PIL import Image
from loader import DataLoader
# from  load_pad_resize import load_pad_resize
plt.ion()   # interactive mode
# /torch.cuda.set_rng_state(torch.ByteTensor(5))
torch.manual_seed(786)
# torch.cuda.manual_seed_all(786)
nclasses = 5  ### the number of classes ........
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)


############### The damn transforms you wish to do ###################################
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([ 0.71691994132405656, 0.61684994868324416, 0.8422594726980035], [ 0.14950700743397516, 0.17600525075050336, 0.090488106486490805]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.RandomHorizontalFlip()
        # transforms.RandomVerticalFlip()
    ]),
    'valid': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(240),
        transforms.ToTensor(),
        # transforms.Normalize([ 0.71691994132405656, 0.61684994868324416, 0.8422594726980035], [ 0.14950700743397516, 0.17600525075050336, 0.090488106486490805])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.71001250387099257,  0.61589000191440846, 0.83586583025971373], [0.14995533531622163, 0.17937717188033223, 0.091176315938062691])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(240),
        transforms.ToTensor(),
        # transforms.Normalize([ 0.71691994132405656, 0.61684994868324416, 0.8422594726980035], [ 0.14950700743397516, 0.17600525075050336, 0.090488106486490805])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.71001250387099257,  0.61589000191440846, 0.83586583025971373], [0.14995533531622163, 0.17937717188033223, 0.091176315938062691])
    ]),

}


data_dir = '../../processed_data' ### path where the data is stored.


###################### The data loading path###############################################



# num_of_samples = 500
# class_sample_count = [134, 20, 136, 76, 49] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
# weights = 1 / torch.Tensor(class_sample_count)
# weights=weights.double()
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,num_of_samples)

image_datasets = {x: custom.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
# image_datasets = {x: custom.ImageFolder(os.path.join(data_dir, x))
#                   for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=5,
    num_workers=4,shuffle=True)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
###################### The data loading path ends here###############################################


#### see the class name here
class_names = image_datasets['train'].classes
print(class_names)



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)  # pause a bit so that plots are updated


# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])



#### see the class name here
class_names = image_datasets['train'].classes
print(class_names)
use_gpu = torch.cuda.is_available()


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     plt.imshow(inp[0])
#     plt.figure()
#     plt.imshow(inp[1])

#     if title is not None:
#         plt.title(title)
#     plt.pause(100)  # pause a bit so that plots are updated


##### get a  sample of the data #######################

inputs, classes = next(iter(dataloaders['train']))
c_inputs = inputs.numpy()


#######################################################


### define weight init#################################
def init_weights(m):

    if type(m) == nn.Conv2d :
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
    if type(m)==nn.ConvTranspose2d:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        # init.constant(m.bias,0.0)
#######################################################


####### define check point saving######################
def checkpoint(model_in, epoch,output_path):
    
    model_out_path = output_path+'/'+"model_epoch_{}.pth".format(epoch)

    check_point_wts = model_in.state_dict()
    model_in.load_state_dict(check_point_wts)
    torch.save(model_in, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    del check_point_wts
    print ('check memory...')
    del model_in


# def checkpoint(model_in, epoch,output_path):
    
#     model_out_path = output_path+'/'+"model_epoch_{}.pth".format(epoch)

#     check_point_wts = model_in.state_dict()
#     model_in.load_state_dict(check_point_wts)
#     return model_in
    # torch.save(model_in, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))
    # del check_point_wts
    # print ('check memory...')
    # del model_in
#######################################################

####### time to train the model, ###################################

"""
the train model expects 1) the model 2) the criterion to be minimized/ maximized, 3) the optimizer, 4) scheduler (learning rate decay)
 5) The number of epochs to run , 6) the output folder to save check points
"""
def train_model(model, optimizer, scheduler, num_epochs=25, output_path='/home/bmi/Desktop/'):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    min_loss = 1000.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            confusion_meter.reset() ### reset the confusion matrix
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                print (labels)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                del _ 

                numpified_labels = labels.cpu().data.numpy()
                w0 = 1.0*((numpified_labels==0).sum())
                w1 = 1.0*((numpified_labels==1).sum())
                w2 = 1.0*((numpified_labels==2).sum())
                w3 = 1.0*((numpified_labels==3).sum())
                w4 = 1.0*((numpified_labels==4).sum())

                w_tot = w0+w1+w2+w3+w4

                wce =  torch.from_numpy(np.asarray([1/w0, 1/w1, 1/w2, 1/w3, 1/w4])).float()

                wce= wce.cuda()

                criterion = nn.CrossEntropyLoss(weight=wce)

                loss = criterion(outputs, labels)
                del outputs

                confusion_meter.add(preds.view(-1),labels.data.view(-1))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            del running_loss
            epoch_acc = running_corrects / dataset_sizes[phase]
            del running_corrects
            del preds,inputs

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model and create a check point
            if phase == 'valid' and epoch_loss < min_loss:

                # print ('achieved_best accuracy,hence saving')
                # best_acc = epoch_acc
                # del best_model_wts
                # best_model_wts = model.state_dict()
                # best_model.save_state_dict(output_path+'/'+"model_epoch_{}.pth".format(epoch))
                # best_epoch = epoch
                # # del best_mode
                # # saved_best_model = checkpoint(model,epoch,output_name)
                # # torch.save(saved_best_model, output_path+'/'+"model_epoch_{}.pth".format(epoch))
                # # del saved_best_model
		scheduler.step(epoch_loss) # lr_scheduler based on validation loss
                print ('achieved_best accuracy,hence saving')
                # best_acc = epoch_acc
		min_loss = epoch_loss
                best_model_wts = model.state_dict()
                checkpoint(model,epoch,output_name)



            print (confusion_meter.conf)


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloders['valid']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return






#### code to be used if you want to use pretrained networks#################################
model_urls['densenet121'] = model_urls['densenet121'].replace('https://', 'http://')
model_ft= torchvision.models.densenet121(pretrained=True)  ### when false only the model architecture will be taken, if set to true then the network will be initilaized with weights learnt by training imagenet
# for param in model_ft.parameters():
#     param.requires_grad = False
# print(model_ft)
num_ftrs= model_ft.classifier.in_features
model_ft.classifier= nn.Linear(num_ftrs,nclasses) ### 4 is number of classes
##############################################################################################

if use_gpu:
    model_ft = model_ft.cuda()

# weights   = torch.from_numpy(np.asarray([1/134.0, 1/20.0, 1/136.0, 1/74.0, 1/49.0])).float()
# weights= weights.cuda()

# criterion = nn.CrossEntropyLoss(weight=weights)

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.0001)

# Decay LR by a factor of 0.01 every 3 epochs
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)



output_name= 'densenet161_EMMA_2'
if not os.path.exists(output_name):
    os.mkdir(output_name)

model_ft = train_model(model_ft, optimizer_ft, exp_lr_scheduler,num_epochs=50, output_path=output_name)

torch.save(model_ft,output_name+'/'+output_name+'.pth')
