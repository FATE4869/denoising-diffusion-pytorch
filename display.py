import numpy as np
import pickle
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.utils.prune as prune
from data import get_metadata, get_dataset, fix_legacy_dict
import matplotlib.pyplot as plt
import torch.nn


def main():
    epochs = [439, 489]
    dataset = 'cifar10'
    Ts = [0, 1, 3, 5, 10, 50, 100, 150, 200, 250]
    pr = 'lamp_0.2_per_iteration'
    if pr == 0:
        images_path = f'sampled_images/{dataset}/dense_model/'
    elif isinstance(pr, str):
        images_path = f'sampled_images/{dataset}/{pr}/'
    else:
        images_path = f'sampled_images/{dataset}/random_pr_{pr}/'
    for epoch in epochs:
        with open(f'{images_path}/epoch_{epoch}_checkpoints.pkl', 'rb') as f:
            checkpoints = pickle.load(f)
        fig = plt.figure(figsize=(12, 3))
        for j, T in enumerate(Ts):
            images = np.array(checkpoints[T].cpu()).transpose(0, 2, 3, 1)[:2]
            # scale from  [-1, 1] to [0, 255]
            images = (127.5 * (images + 1)).astype(np.uint8)
            plt.subplot(2, len(Ts),j + 1)
            plt.imshow(images[0])
            plt.axis('off')
            plt.title(f'{T}')
            plt.subplot(2, len(Ts), j + len(Ts) + 1)
            plt.imshow(images[1])
            plt.axis('off')
            fig.suptitle(f'generated images at epoch {epoch}')
        # plt.savefig(f"sampled_images/mnist/random_pr_0.5/epoch_{epoch}_checkpoints.png")
        plt.savefig(f"{images_path }/epoch_{epoch}_checkpoints.png")
        plt.show()

def show_images(epoch, num_images=2048, dataset='cifar10'):
    images = torch.load(f'sampled_images/{dataset}/dense/epoch_{epoch}_Unet_{dataset}-250-sampling_steps-{num_images}_images-class_condn_False.pt')
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    plt.figure(figsize=(10, 8))
    counter = 1
    num_col = 4
    num_row = 3
    for i in range(num_col):
        for j in range(num_row):
            plt.subplot(num_row, num_col, counter)
            plt.imshow(images[counter-1])
            counter += 1
    plt.savefig(f"sampled_images/{dataset}/dense/epoch_{epoch}_images.png")
    plt.show()

def add_channels(data, target_channel):
    org_data = data
    while data.shape[1] < target_channel:
        data = torch.cat((data, org_data), dim = 1)
    return data

def fid_score(epoch):
    # dataset = 'mnist'
    dataset = 'CIFAR10'
    metadata = get_metadata(dataset.lower())
    data_dir = './dataset /'
    train_set = get_dataset(dataset.lower(), data_dir, metadata)
    if dataset == 'CIFAR10':
        train_data = torch.from_numpy(train_set.data.transpose(0, 3, 1, 2))
    # elif dataset == 'mnist':
        # num_samples * H * W -> num_samples * 1 * H * W
        # train_data = torch.unsqueeze(train_set.train_data, dim=1)
        # train_data = add_channels(train_data, target_channel=3)
    print(train_data.shape)

    # epoch = 489
    # pr = 'lamp_0.2_per_iteration'
    # if pr == 0:
    data_load_path = f'sampled_images/{dataset}/dense/epoch_500_Unet_cifar10-250-sampling_steps-2048_images-class_condn_False.pt'
    # elif isinstance(pr, str):
    #     data_load_path = f'sampled_images/{dataset}/{pr}/arr_0_epoch_{epoch}_images_2048.npy'
    # else:
    #     data_load_path = f'sampled_images/{dataset}/random_pr_{pr}/arr_0_epoch_{epoch}_images_2048.npy'
    #
    # # data_load_path = 'sampled_images/mnist/random_pr_0.5/arr_0_epoch_29_images_1024.npy'
    generated_data = torch.load(data_load_path)
    print(generated_data.shape)
    generated_data.transpose(1, 3).transpose(2, 3)
    print(generated_data.shape)
    # print(f'Loading generated data from: {data_load_path}')
    # # num_samples * H * W * C -> num_samples * C * H * W
    # generated_data = torch.tensor(generated_data.transpose(0, 3, 1, 2))
    #
    # # num_samples * 1 *H * W -> num_samples * 3 * H * W
    # generated_data = add_channels(generated_data, target_channel=3)
    # # generated_data = torch.cat((generated_data, generated_data, generated_data), dim=1)
    # print(generated_data.shape)
    #
    fid = FrechetInceptionDistance(feature=64)
    fid.update(train_data[:2048], real=True)
    fid.update(generated_data, real=False)
    print(f'FID score: {fid.compute():.2f}')


def param_sum():
    dataset = 'mnist'
    metadata = get_metadata(dataset)
    class_cond = True
    # Creat model and diffusion process
    model = unets.__dict__['UNet'](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if class_cond else None,
    ).to('cuda')

    input_blocks_sum_params = torch.zeros(16)
    for i, (name, param) in enumerate(model.input_blocks.named_parameters()):
        # print(name)
        cur_layer = int(name.split('.')[0])
        input_blocks_sum_params[cur_layer] += param.numel()
    print(input_blocks_sum_params)
    print(torch.sum(input_blocks_sum_params))

    middle_blocks_sum_params = torch.zeros(3)
    for i, (name, param) in enumerate(model.middle_block.named_parameters()):
        # print(name)
        cur_layer = int(name.split('.')[0])
        middle_blocks_sum_params[cur_layer] += param.numel()
    print(middle_blocks_sum_params)
    print(torch.sum(middle_blocks_sum_params))

    output_blocks_sum_params = torch.zeros(16)
    for i, (name, param) in enumerate(model.output_blocks.named_parameters()):
        # print(name)
        cur_layer = int(name.split('.')[0])
        output_blocks_sum_params[cur_layer] += param.numel()
    print(output_blocks_sum_params)
    print(torch.sum(output_blocks_sum_params))

    sum_params = torch.cat((input_blocks_sum_params, middle_blocks_sum_params, output_blocks_sum_params))
    print(sum_params)
    plt.figure()
    plt.scatter(torch.arange(0, 16), sum_params[0:16], color='g', label='input blocks')
    plt.scatter(torch.arange(16, 19), sum_params[16:19], color='r', label='mid blocks')
    plt.scatter(torch.arange(19, 35), sum_params[19:35], color='b', label='output blocks')
    plt.ylabel('num of param')
    plt.xticks([0, 15, 18, 34])
    # plt.xticks([7, 17, 27], ['input blocks',  'mid block', 'out blocks'])
    plt.title("Number of parameters in each block")
    plt.legend()
    plt.show()
    print("done")


def prune_param():
    dataset = 'mnist'
    metadata = get_metadata(dataset)
    class_cond = True
    # Creat model and diffusion process
    model = unets.__dict__['UNet'](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if class_cond else None,
    ).to('cuda')

    count = 0
    sums = []
    for i, (name, module) in enumerate(model.input_blocks.named_modules()):
        # for name, param in model.named_parameters():
        sum = 0
        if isinstance(module, unets.TimestepEmbedSequential):
            for inner_name, inner_module in module.named_modules():
                # sum += inner_module.numel()
                if isinstance(inner_module, torch.nn.Conv2d):
                    prune.random_unstructured(inner_module, name='weight', amount=0.2)
                    prune.random_unstructured(inner_module, name='bias', amount=0.2)
                    print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of weights: {torch.numel(inner_module.weight)}')
                    print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of bias: {torch.numel(inner_module.bias)}')
                    count += 2
                elif isinstance(inner_module, torch.nn.Linear):
                    prune.random_unstructured(inner_module, name='weight', amount=0.2)
                    prune.random_unstructured(inner_module, name='bias', amount=0.2)
                    print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of weights: {torch.numel(inner_module.weight)}')
                    print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of bias: {torch.numel(inner_module.bias)}')
                    count += 2
                elif isinstance(inner_module, unets.GroupNorm32):
                    prune.random_unstructured(inner_module, name='weight', amount=0.2)
                    prune.random_unstructured(inner_module, name='bias', amount=0.2)
                    print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of weights: {torch.numel(inner_module.weight)}')
                    print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of bias: {torch.numel(inner_module.bias)}')
                    count += 2
            sums.append(sum)
    print(sums)
    print(count)

if __name__ == '__main__':
    # main()
    show_images(epoch=500)
    # epochs = [499]
    # for epoch in epochs:
    #     fid_score(epoch)
    # param_sum()
    # prune_param()