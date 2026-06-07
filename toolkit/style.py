from torch import nn
import torch
from torchvision import models

def tensor_size(tensor):
    return tensor.shape[1] * tensor.shape[2] * tensor.shape[3]

class ContentLoss(nn.Module):
    def __init__(self, single_target=False):
        super(ContentLoss, self).__init__()
        self.single_target = single_target
        self.loss = None

    def forward(self, stacked_input):
        if self.single_target:
            split_size = stacked_input.size(0) // 2
            pred_layer, target_layer = torch.split(stacked_input, split_size, dim=0)
        else:
            split_size = stacked_input.size(0) // 3
            pred_layer, _, target_layer = torch.split(stacked_input, split_size, dim=0)

        content_size = tensor_size(pred_layer)

        # Вычисляем математику напрямую, без тормозных вложенных функций
        diff = torch.abs(pred_layer.float() - target_layer.float())
        l2 = torch.sum(diff ** 2, dim=[1, 2, 3], keepdim=True) / 2.0
        pred_itemized_loss = 2. * l2 / content_size

        if torch.isnan(pred_itemized_loss).any():
            print('pred_itemized_loss is nan')

        self.loss = torch.mean(pred_itemized_loss, dim=(1, 2, 3), keepdim=True)
        return stacked_input

def convert_to_gram_matrix(inputs):
    inputs = inputs.float()
    batch, filters, height, width = inputs.size()
    size = height * width * filters

    feats = inputs.view(batch, filters, height * width)
    feats_t = feats.transpose(1, 2)
    # Используем bmm (Batch Matrix-Matrix) - работает быстрее стандартного matmul
    grams_raw = torch.bmm(feats, feats_t) 
    return grams_raw / size

class StyleLoss(nn.Module):
    def __init__(self, single_target=False):
        super(StyleLoss, self).__init__()
        self.single_target = single_target

    def forward(self, stacked_input):
        input_dtype = stacked_input.dtype
        stacked_input_f = stacked_input.float()
        
        if self.single_target:
            split_size = stacked_input_f.size(0) // 2
            preds, style_target = torch.split(stacked_input_f, split_size, dim=0)
        else:
            split_size = stacked_input_f.size(0) // 3
            preds, style_target, _ = torch.split(stacked_input_f, split_size, dim=0)

        target_grams = convert_to_gram_matrix(style_target)
        pred_grams = convert_to_gram_matrix(preds)
        
        gram_size = target_grams.size(1) * target_grams.size(2)
        diff = torch.abs(pred_grams - target_grams)
        itemized_loss = torch.sum(diff ** 2, dim=(1, 2), keepdim=True) / gram_size

        if torch.isnan(itemized_loss).any():
            print('itemized_loss is nan')
            
        itemized_loss = torch.unsqueeze(itemized_loss, dim=1)
        self.loss = torch.mean(itemized_loss, dim=(1, 2), keepdim=True).to(input_dtype)
        return stacked_input

class Normalization(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(Normalization, self).__init__()
        # Правильный PyTorch-way: регистрируем константы как буферы модели. 
        # Они не требуют градиентов и автоматически прыгают в видеопамять.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))
        self.dtype = dtype

    def forward(self, stacked_input):
        if stacked_input.shape[1] == 4:
            stacked_input = stacked_input[:, :3, :, :]
        return ((stacked_input - self.mean) / self.std).to(self.dtype)

class OutputLayer(nn.Module):
    def __init__(self, name='output_layer'):
        super(OutputLayer, self).__init__()
        self.name = name

    def forward(self, stacked_input):
        return stacked_input

def get_style_model_and_losses(
        single_target=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_layer_name=None,
        dtype=torch.float32
):
    content_layers = ['conv2_2', 'conv3_2', 'conv4_2']
    style_layers = ['conv2_1', 'conv3_1', 'conv4_1']
    
    # Загружаем VGG19
    cnn = models.vgg19(pretrained=True).features.to(device, dtype=dtype).eval()
    
    # === КРИТИЧЕСКИЙ ФИКС ===
    # Жестко отключаем градиенты для всех слоев VGG19!
    # Это спасает видеокарту от пустых вычислений и экономит прорву VRAM.
    for param in cnn.parameters():
        param.requires_grad = False

    normalization = Normalization(dtype=dtype).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    block = 1
    output_layer = None

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv{block}_{i}_raw'
        elif isinstance(layer, nn.ReLU):
            name = f'conv{block}_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
            block += 1
            i = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            content_loss = ContentLoss(single_target=single_target).to(device)
            model.add_module(f"content_loss_{block}_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            style_loss = StyleLoss(single_target=single_target).to(device)
            model.add_module(f"style_loss_{block}_{i}", style_loss)
            style_losses.append(style_loss)

        if output_layer_name is not None and name == output_layer_name:
            output_layer = OutputLayer(name)
            model.add_module(f"output_layer_{block}_{i}", output_layer)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss, OutputLayer)):
            break

    model = model[:(i + 1)].to(dtype=dtype)

    return model, style_losses, content_losses, output_layer