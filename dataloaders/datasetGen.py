import torch
from random import shuffle
from .wrapper import Subclass, AppendName, Permutation
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset

def SplitGen(train_dataset, val_dataset, first_split_sz=2, other_split_sz=2, rand_split=False, remap_class=False):
    '''
    Generate the dataset splits based on the labels.
    :param train_dataset: (torch.utils.data.dataset)
    :param val_dataset: (torch.utils.data.dataset)
    :param first_split_sz: (int)
    :param other_split_sz: (int)
    :param rand_split: (bool) Randomize the set of label in each split
    :param remap_class: (bool) Ex: remap classes in a split from [2,4,6 ...] to [0,1,2 ...]
    :return: train_loaders {task_name:loader}, val_loaders {task_name:loader}, out_dim {task_name:num_classes}
    '''
    assert train_dataset.number_classes==val_dataset.number_classes,'Train/Val has different number of classes'
    num_classes =  train_dataset.number_classes

    # Calculate the boundary index of classes for splits
    # Ex: [0,2,4,6,8,10] or [0,50,60,70,80,90,100]
    split_boundaries = [0, first_split_sz]
    while split_boundaries[-1]<num_classes:
        split_boundaries.append(split_boundaries[-1]+other_split_sz)
    print('split_boundaries:',split_boundaries)
    assert split_boundaries[-1]==num_classes,'Invalid split size'

    # Assign classes to each splits
    # Create the dict: {split_name1:[2,6,7], split_name2:[0,3,9], ...}
    if not rand_split:
        class_lists = {str(i):list(range(split_boundaries[i-1],split_boundaries[i])) for i in range(1,len(split_boundaries))}
    else:
        randseq = torch.randperm(num_classes)
        class_lists = {str(i):randseq[list(range(split_boundaries[i-1],split_boundaries[i]))].tolist() for i in range(1,len(split_boundaries))}
    print(class_lists)

    # Generate the dicts of splits
    # Ex: {split_name1:dataset_split1, split_name2:dataset_split2, ...}
    train_dataset_splits = {}
    val_dataset_splits = {}
    task_output_space = {}
    for name,class_list in class_lists.items():
        train_dataset_splits[name] = AppendName(Subclass(train_dataset, class_list, remap_class), name)
        val_dataset_splits[name] = AppendName(Subclass(val_dataset, class_list, remap_class), name)
        task_output_space[name] = len(class_list)

    return train_dataset_splits, val_dataset_splits, task_output_space


def PermutedGen(train_dataset, val_dataset, n_permute, remap_class=False):
    sample, _ = train_dataset[0]
    n = sample.numel()
    train_datasets = {}
    val_datasets = {}
    task_output_space = {}
    for i in range(1,n_permute+1):
        rand_ind = list(range(n))
        shuffle(rand_ind)
        name = str(i)
        if i==1:  # First task has no permutation
            train_datasets[name] = AppendName(train_dataset, name)
            val_datasets[name] = AppendName(val_dataset, name)
        else:
            # For incremental class scenario, use remap_class=True
            first_class_ind = (i-1)*train_dataset.number_classes if remap_class else 0
            train_datasets[name] = AppendName(Permutation(train_dataset, rand_ind), name, first_class_ind=first_class_ind)
            val_datasets[name] = AppendName(Permutation(val_dataset, rand_ind), name, first_class_ind=first_class_ind)
        task_output_space[name] = train_dataset.number_classes

    return train_datasets, val_datasets, task_output_space

def CORe50Gen():
    _mu = [0.485, 0.456, 0.406]  # imagenet normalization
    _std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mu, std=_std)
    ])
    val_transform = train_transform
    
    train_datasets = {}
    val_datasets = {}
    train_task_output_space = {}
    val_task_output_space = {}

    train_data1 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s1/',transform=train_transform)
    train_data2 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s2/',transform=train_transform)
    train_data3 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s3/',transform=train_transform)
    train_data4 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s4/',transform=train_transform)
    train_data5 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s5/',transform=train_transform)
    train_data6 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s6/',transform=train_transform)
    train_data7 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s7/',transform=train_transform)
    train_data8 = ImageFolder(root='/data1/zhaohongbo/core50_128x128/s8/',transform=train_transform)

    train_datasets['1']=AppendName(train_data1,str(1))
    train_datasets['2']=AppendName(train_data2,str(2))
    train_datasets['3']=AppendName(train_data3,str(3))
    train_datasets['4']=AppendName(train_data4,str(4))   
    train_datasets['5']=AppendName(train_data5,str(5))
    train_datasets['6']=AppendName(train_data6,str(6))
    train_datasets['7']=AppendName(train_data7,str(7))
    train_datasets['8']=AppendName(train_data8,str(8))

    # val dataset
    val_data1 =  ImageFolder(root='/data1/zhaohongbo/core50_128x128/s9/',transform=val_transform)  
    val_data2 =  ImageFolder(root='/data1/zhaohongbo/core50_128x128/s10/',transform=val_transform)  
    val_data3 =  ImageFolder(root='/data1/zhaohongbo/core50_128x128/s11/',transform=val_transform)  

    val_datasets['1'] = AppendName(val_data1,str(1))
    val_datasets['2'] = AppendName(val_data2,str(2))
    val_datasets['3'] = AppendName(val_data3,str(3))

    for i in range(1,9):
        train_task_output_space[str(i)]=50
    
    for i in range(1,4):
        val_task_output_space[(str(i))]=50

    return train_datasets, val_datasets, train_task_output_space,val_task_output_space

def datasets4Gen(dataroot):
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    val_transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    train_data1 = ImageFolder(root=dataroot+'cubs_cropped/train',transform=train_transform)
    train_data2 =ImageFolder(root=dataroot+'flowers/train',transform=train_transform)
    train_data3 =ImageFolder(root=dataroot+'sketches/train',transform=train_transform)
    train_data4 =ImageFolder(root=dataroot+'stanford_cars_cropped/train',transform=train_transform)
    
    val_data1 = ImageFolder(root=dataroot+'cubs_cropped/test',transform=val_transform)
    val_data2 =ImageFolder(root=dataroot+'flowers/test',transform=val_transform)
    val_data3 =ImageFolder(root=dataroot+'sketches/test',transform=val_transform)
    val_data4 =ImageFolder(root=dataroot+'stanford_cars_cropped/test',transform=val_transform)

    train_datasets = {}
    val_datasets = {}
    train_task_output_space = {}
    val_task_output_space = {}

    train_datasets['1']=AppendName(train_data1,str(1))
    train_datasets['2']=AppendName(train_data2,str(2))
    train_datasets['3']=AppendName(train_data3,str(3))
    train_datasets['4']=AppendName(train_data4,str(4))          

    val_datasets['1'] = AppendName(val_data1,str(1))
    val_datasets['2'] = AppendName(val_data2,str(2))
    val_datasets['3'] = AppendName(val_data3,str(3))
    val_datasets['4'] = AppendName(val_data4,str(4))

    train_task_output_space['1'] = 200 # cubs
    train_task_output_space['2'] = 102 # flowers
    train_task_output_space['3'] = 250 # sketches
    train_task_output_space['4'] = 196 # cars

    return train_datasets, val_datasets, train_task_output_space

def officehomeGen(dataroot,fac=0.7,seed=0):
    train_datasets = {}
    val_datasets = {}
    train_task_output_space = {}

    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    
    data1 = ImageFolder(root=dataroot+'Art')
    data2 = ImageFolder(root=dataroot+'Clipart')
    data3 = ImageFolder(root=dataroot+'Product')
    data4 = ImageFolder(root=dataroot+'Real World')

    train1_size=int(len(data1)*fac)
    val1_size=len(data1)-train1_size
    train2_size=int(len(data2)*fac)
    val2_size=len(data2)-train2_size
    train3_size=int(len(data3)*fac)
    val3_size=len(data3)-train3_size
    train4_size=int(len(data4)*fac)
    val4_size=len(data4)-train4_size

    train_data1,val_data1 = random_split(dataset=data1,lengths=[train1_size,val1_size], generator=torch.Generator().manual_seed(seed))
    train_data2,val_data2 =random_split(dataset=data2,lengths=[train2_size,val2_size], generator=torch.Generator().manual_seed(seed))
    train_data3,val_data3 =random_split(dataset=data3,lengths=[train3_size,val3_size], generator=torch.Generator().manual_seed(seed))
    train_data4,val_data4 =random_split(dataset=data4,lengths=[train4_size,val4_size], generator=torch.Generator().manual_seed(seed))

    train_data1=MyLazyDataset(train_data1.dataset, transform['train'])
    train_data2=MyLazyDataset(train_data2.dataset, transform['train'])
    train_data3=MyLazyDataset(train_data3.dataset, transform['train'])
    train_data4=MyLazyDataset(train_data4.dataset, transform['train'])

    val_data1=MyLazyDataset(val_data1.dataset,transform['test'])
    val_data2=MyLazyDataset(val_data2.dataset,transform['test'])
    val_data3=MyLazyDataset(val_data3.dataset,transform['test'])
    val_data4=MyLazyDataset(val_data4.dataset,transform['test'])

    train_datasets['1']=AppendName(train_data1,str(1))
    train_datasets['2']=AppendName(train_data2,str(2))
    train_datasets['3']=AppendName(train_data3,str(3))
    train_datasets['4']=AppendName(train_data4,str(4))          

    val_datasets['1'] = AppendName(val_data1,str(1))
    val_datasets['2'] = AppendName(val_data2,str(2))
    val_datasets['3'] = AppendName(val_data3,str(3))
    val_datasets['4'] = AppendName(val_data4,str(4))

    train_task_output_space['1'] = 65
    train_task_output_space['2'] = 65
    train_task_output_space['3'] = 65
    train_task_output_space['4'] = 65
    
    return train_datasets, val_datasets, train_task_output_space

class MyLazyDataset(Dataset): # train, test with different transforms
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)