# A-demo-to-run-Swin-Transformer-and-Token-Clustering-Transformer
The original code of Swin Transformer and Token Clustering Transformer haven't been repaired for a long time and are no longer able to run since they  both used the mm-lab pipeline, which have experienced significant change. This code provide an example how to run the two model under new criterion. 

## Get Started
## Code Link
```shell
https://disk.pku.edu.cn/link/AAABD7E5FE06D94E81BBB30223F9FED2ED
Name: TCFormer_n.zip
Expires: Never
Pickup Code: 1234
```
### Env Configuration
```shell
mmcv                          2.1.0
mmdet                         3.3.0
mmengine                      0.10.4
mmpose                        1.3.2               
mmsegmentation                1.2.2
numpy                         1.26.3
timm                          1.0.7
torch                         2.0.0+cu118
torchvision                   0.15.1+cu118
```
See others in EnvInfo.txt
### How to run the code of trainning
After configuring the env, pay attention to the file /TCFormer_n/tools/train.py, first you should run the command:
```shell
conda activate nat  # or the env name you created for this pipeline
```
then you should change the value of '--config' and '--work-dir' to the path of the config file of the model you want to train and the work directory you want to restore the model and log file respectively.
And you can change the args in the config files according to your own need when necessary.
The config files are under the path of:
```shell
/TCFormer_n/configs/
```
As we use the ADE2016Challenge as data, the dataset config files are in:
```shell
/TCFormer_n/configs/_base_/datasets/ade20k.py
```
where you can change the scale arguments according to your own need. Here I use (512, 512) which is (512, 683) in raw img and (2048, 512) in the raw code provided by mmseg official.
And you can adjust your trainning process by change arguments in shedule configuration files.
After configuring, you can just run:
```shell
python train.py
```
to train the model, and the trainning log and checkpoint will be saved in the work directory you set. The checkpoint will be saved and the model will be validated all according to the intervals you set in shedule files in:
```shell
/TCFormer_n/configs/_base_/schedules
```
### Testing the model
After trainning, the checkpoint files are saved in the work directory you just set, and turn to the file test.py which is in the same directory as train.py .
You need to change the args below:
```shell
'--config' # just the same as trainning stage
'--checkpoint' # the checkpoint you want to load and test saved from trainning stage
'--work-dir' # 'if specified, the evaluation metric results will be dumped into the directory as json'
'--out' # 'The directory to save output prediction for offline evaluation'
'--show' # 'show prediction results'
'--show-dir' # 'directory where painted images will be saved. If specified, it will be automatically saved to the work_dir/timestamp/show_dir'
'--tta' # 'Test time augmentation'
```
After resetting the arguments, you can run command:
```shell
test.py
```
to test the model.

# References
```latex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
@inproceedings{zeng2022not,
  title={Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer},
  author={Zeng, Wang and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11101--11111},
  year={2022}
}

@article{zeng2024tcformer,
  title={TCFormer: Visual Recognition via Token Clustering Transformer},
  author={Zeng, Wang and Jin, Sheng and Xu, Lumin and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping and Wang, Xiaogang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
