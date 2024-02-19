
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from geofree.util import get_local_path, OmegaConf
from geofree.main import instantiate_from_config
import torchvision.transforms as transforms
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
import copy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    """
    A gram matrix is the result of multiplying a given matrix by its transposed matrix. 
    """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    

class StyleTransfer:
    def __init__(self, style_weight=1000000, content_weight=1):
        self.cnn = models.vgg19(pretrained=True).features.to(device).eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(self,
                               style_img, content_img):
        cnn = copy.deepcopy(self.cnn)

        # normalization module
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
    
    def filter_object(self,content_image,output,seg_map,selected_object):
        mask = np.zeros_like(seg_map.cpu().squeeze(0).squeeze(0).numpy())
        mask[seg_map.cpu().squeeze(0).squeeze(0).numpy() == selected_object] = 1

        # Make the mask 3D
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)


        final_output = output.detach().cpu().clone().squeeze(0).permute(1, 2, 0)
        final_output = np.asarray(final_output)
        final_output = final_output * mask

        #Add the original image to the output image
        remaining_image = content_image.cpu().clone().squeeze(0).permute(1, 2, 0).numpy() * (1 - mask)

        final_output = final_output + remaining_image

        return final_output

    def train(self, content_img, style_img, seg_map, selected_object,num_steps=200):
        """Run the style transfer. only on the the object we want to transfer the style to"""
        print('Building the style transfer model..')
        content_img = content_img.clone().to(device)
        style_img = style_img.to(device)
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)


        input_img = content_img.clone().to(device)
        
        optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

        print('Optimizing..')
        epoch = [0]
        images = []
        while epoch[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= self.style_weight
                content_score *= self.content_weight

                loss = style_score + content_score
                loss.backward()

                epoch[0] += 1
                if epoch[0] % 50 == 0:
                    print("epoch {}:".format(epoch))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                if epoch[0] % 10 == 0 or epoch[0] < 10:
                    img = input_img.cpu().clone().clamp_(0, 1)
                    img = img.squeeze(0)      # remove the fake batch dimension
                    img = transforms.ToPILImage()(img)
                    images.append(img)
                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return self.filter_object(content_img,input_img,seg_map,selected_object), images

