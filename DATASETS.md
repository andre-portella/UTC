# Download the Datasets
Place all datasets under the same folder (called $DATA) to make management easier, and follow the instructions below to organize them so you avoid having to modify the source code. The file structure should follow this pattern:

```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– dtd/
```

The datasets and their respective splits can be downloaded from this [link](https://drive.google.com/file/d/1MDbcoMdApFPF4SNMyRc9v8Yp6VDW2_Ze/view?usp=drive_link).

List of datasets:

- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [Flowers102](#flowers102)
- [FGVCAircraft](#fgvcaircraft)
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- [CIFAR-10](#cifar10_custom)
- [STL-10](#cifar10_custom)

Instructions for preparing each dataset are detailed below. To ensure reproducibility and a fair comparison in future work, we provide fixed train/val/test splits for all datasets. The fixed splits come from the original datasets or were created by us.

### Caltech101
- Create a folder named `caltech-101/` inside `$DATA`.
- Download `101_ObjectCategories.tar.gz` from [link](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz) and extract it into `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1pfTLKC1MtHe84CHo-VeS1D8mNkdqtaB2/view?usp=drive_link) and place it in `$DATA/caltech-101`.

The directory structure should look like this:
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets
- Create a folder named `oxford_pets/` inside `$DATA`.
- Download the images from [link](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz).
- Download the annotations from [link](https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz).
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1E59ZB1-tEJO0HXUSTZQzER4zt3ZrXGIH/view?usp=drive_link).

The directory structure should look like this:
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### Flowers102
- Create a folder named `oxford_flowers/` inside `$DATA`.
- Download the images and labels from [link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) and [link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) respectively.
- Download `cat_to_name.json` from [link](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing).
- Download `split_zhou_OxfordFlowers.json` from this [link](https://drive.google.com/file/d/118yF_xkSHZ_kQRjGCuH6SoU_lrtHqOHd/view?usp=drive_link).

The directory structure should look like this:
```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### FGVCAircraft
- Download the data from [link](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz).
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like this:
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### DTD
- Download the dataset from [link](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) and extract it into `$DATA`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1IO0D-ZmEvlag56_u-SuQpMxx0ufh2zz-/view?usp=drive_link).

The directory structure should look like this:
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` inside `$DATA`.
- Download the dataset from [link](https://zenodo.org/records/7711810) and extract it into `$DATA/eurosat/`.
- Download `split_zhou_EuroSAT.json` from this [link](https://drive.google.com/file/d/1YgFd15Ra1wz9p7PDz4aR6ra7Lp1g4AfQ/view?usp=drive_link).

The directory structure should look like this:
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### CIFAR-10
- Create a folder named `cifar10/` inside `$DATA`.
- The CIFAR-10 download should be handled by `torchvision.datasets.CIFAR10`.
- Download `split_zhou_CIFAR10.json` from this [link](https://drive.google.com/file/d/1gLJyDsyt_A5bcxI8IU8wrH0XMyRKLKV_/view?usp=drive_link).

The directory structure should look like this:
```
cifar10/
|–– images/
|–– split_zhou_CIFAR10.json
```

### STL-10
- Create a folder named `stl10/` inside `$DATA`.
- The STL-10 download should be handled by `torchvision.datasets.STL10`.
- Download `split_zhou_STL10.json` from this [link](https://drive.google.com/file/d/1PdGx5KIwErP-MNJGtLxePJnAKN2cHSOE/view?usp=drive_link).

The directory structure should look like this:
```
stl10/
|–– images/
|–– split_zhou_STL10.json
```