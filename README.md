This is my complete process of configuring the environment, I hope it can be helpful to you
### 1. programming environment setup with cuda: 12.2

```
conda create -n lapaTemp python=3.10 -y
conda activate lapaTemp

```
Then clone the code:
`git clone https://github.com/LatentActionPretraining/LAPA.git
`
Remove the jax installation from the original project. The new documents `requirement.txt` are as follows:
```
git+https://github.com/lhao499/tux.git
flax==0.7.0
optax==0.1.7
chex==0.1.82
einops
transformers==4.29.2
datasets==2.13.0
tqdm
ml_collections
wandb
gcsfs
requests
typing-extensions
sentencepiece
Pillow
ipdb
imageio[ffmpeg]
decord
tiktoken
tensorflow[and-cuda]
scipy==1.12.0
albumentations
uvicorn
fastapi
```
execution instruction
```pip install -r requirements.txt ```
Then installation jax/jaxlib:
```pip install --upgrade jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```
At this point, I get the following error:

> (lapaTemp) lyd@zkyd:/data/lyd/workspace/LAPA$ pip install --upgrade jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
> Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
> Looking in links: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
> Collecting jax==0.4.23
>   Using cached https://pypi.tuna.tsinghua.edu.cn/packages/28/d0/edf653ea02628f2130ea2557f96d02b264768a2f54d22a9c002c7119cb1d/jax-0.4.23-py3-none-any.whl (1.7 MB)
> Collecting jaxlib==0.4.23+cuda12.cudnn89
>   Using cached https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.23%2Bcuda12.cudnn89-cp310-cp310-manylinux2014_x86_64.whl (131.8 MB)
> Requirement already satisfied: ml-dtypes>=0.2.0 in /data/lyd/miniconda3/envs/lapaTemp/lib/python3.10/site-packages (from jax==0.4.23) (0.5.1)
> Requirement already satisfied: numpy>=1.22 in /data/lyd/miniconda3/envs/lapaTemp/lib/python3.10/site-packages (from jax==0.4.23) (1.26.4)
> Requirement already satisfied: opt-einsum in /data/lyd/miniconda3/envs/lapaTemp/lib/python3.10/site-packages (from jax==0.4.23) (3.4.0)
> Requirement already satisfied: scipy>=1.9 in /data/lyd/miniconda3/envs/lapaTemp/lib/python3.10/site-packages (from jax==0.4.23) (1.12.0)
> Installing collected packages: jaxlib, jax
>   Attempting uninstall: jaxlib
>     Found existing installation: jaxlib 0.5.3
>     Uninstalling jaxlib-0.5.3:
>       Successfully uninstalled jaxlib-0.5.3
>   Attempting uninstall: jax
>     Found existing installation: jax 0.5.3
>     Uninstalling jax-0.5.3:
>       Successfully uninstalled jax-0.5.3
> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
> orbax-checkpoint 0.11.10 requires jax>=0.5.0, but you have jax 0.4.23 which is incompatible.
> Successfully installed jax-0.4.23 jaxlib-0.4.23+cuda12.cudnn89


The orbax-checkpoint 0.11.10 version conflicts with the jax version. To reduce the orbax-checkpoint version, run the following command:
```pip install orbax-checkpoint==0.5.10```

Install jax/jaxlib again:
```pip install --upgrade jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```

The installation is successful. If the error still occurs, the specific version requirements will be prompted
Finally, install pytorch:
``` pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118```
Finished

**Here are the versions of all the environment variables**
> # Name                    Version                   Build  Channel
> _libgcc_mutex             0.1                        main  
> _openmp_mutex             5.1                       1_gnu  
> absl-py                   2.2.1                    pypi_0  
> aiohappyeyeballs          2.6.1                    pypi_0  
> aiohttp                   3.11.16                  pypi_0  
> aiosignal                 1.3.2                    pypi_0  
> albucore                  0.0.23                   pypi_0  
> albumentations            2.0.5                    pypi_0  
> annotated-types           0.7.0                    pypi_0  
> anyio                     4.9.0                    pypi_0  
> asttokens                 3.0.0                    pypi_0  
> astunparse                1.6.3                    pypi_0  
> async-timeout             5.0.1                    pypi_0  
> attrs                     25.3.0                   pypi_0  
> bzip2                     1.0.8                h5eee18b_6  
> ca-certificates           2025.2.25            h06a4308_0  
> cachetools                5.5.2                    pypi_0  
> certifi                   2025.1.31                pypi_0  
> charset-normalizer        3.4.1                    pypi_0  
> chex                      0.1.82                   pypi_0  
> click                     8.1.8                    pypi_0  
> cloudpickle               3.1.1                    pypi_0  
> datasets                  2.13.0                   pypi_0  
> decorator                 5.2.1                    pypi_0  
> decord                    0.6.0                    pypi_0  
> dill                      0.3.6                    pypi_0  
> docker-pycreds            0.4.0                    pypi_0  
> einops                    0.8.1                    pypi_0  
> etils                     1.12.2                   pypi_0  
> exceptiongroup            1.2.2                    pypi_0  
> executing                 2.2.0                    pypi_0  
> fastapi                   0.115.12                 pypi_0  
> filelock                  3.18.0                   pypi_0  
> flatbuffers               25.2.10                  pypi_0  
> flax                      0.7.0                    pypi_0  
> frozenlist                1.5.0                    pypi_0  
> fsspec                    2025.3.2                 pypi_0  
> gast                      0.6.0                    pypi_0  
> gcsfs                     2025.3.2                 pypi_0  
> gitdb                     4.0.12                   pypi_0  
> gitpython                 3.1.44                   pypi_0  
> google-api-core           2.24.2                   pypi_0  
> google-auth               2.38.0                   pypi_0  
> google-auth-oauthlib      1.2.1                    pypi_0  
> google-cloud-core         2.4.3                    pypi_0  
> google-cloud-storage      3.1.0                    pypi_0  
> google-crc32c             1.7.1                    pypi_0  
> google-pasta              0.2.0                    pypi_0  
> google-resumable-media    2.7.2                    pypi_0  
> googleapis-common-protos  1.69.2                   pypi_0  
> grpcio                    1.71.0                   pypi_0  
> h11                       0.14.0                   pypi_0  
> h5py                      3.13.0                   pypi_0  
> huggingface-hub           0.30.1                   pypi_0  
> humanize                  4.12.2                   pypi_0  
> idna                      3.10                     pypi_0  
> imageio                   2.37.0                   pypi_0  
> imageio-ffmpeg            0.6.0                    pypi_0  
> importlib-resources       6.5.2                    pypi_0  
> ipdb                      0.13.13                  pypi_0  
> ipython                   8.34.0                   pypi_0  
> jax                       0.4.23                   pypi_0  
> jaxlib                    0.4.23+cuda12.cudnn89          pypi_0  
> jedi                      0.19.2                   pypi_0  
> jinja2                    3.1.4                    pypi_0  
> keras                     3.9.2                    pypi_0  
> ld_impl_linux-64          2.40                 h12ee557_0  
> libclang                  18.1.1                   pypi_0  
> libffi                    3.4.4                h6a678d5_1  
> libgcc-ng                 11.2.0               h1234567_1  
> libgomp                   11.2.0               h1234567_1  
> libstdcxx-ng              11.2.0               h1234567_1  
> libuuid                   1.41.5               h5eee18b_0  
> markdown                  3.7                      pypi_0  
> markdown-it-py            3.0.0                    pypi_0  
> markupsafe                3.0.2                    pypi_0  
> matplotlib-inline         0.1.7                    pypi_0  
> mdurl                     0.1.2                    pypi_0  
> ml-collections            1.0.0                    pypi_0  
> ml-dtypes                 0.5.1                    pypi_0  
> mpmath                    1.3.0                    pypi_0  
> msgpack                   1.1.0                    pypi_0  
> multidict                 6.2.0                    pypi_0  
> multiprocess              0.70.14                  pypi_0  
> namex                     0.0.8                    pypi_0  
> ncurses                   6.4                  h6a678d5_0  
> nest-asyncio              1.6.0                    pypi_0  
> networkx                  3.3                      pypi_0  
> numpy                     1.26.4                   pypi_0  
> nvidia-cublas-cu11        11.11.3.6                pypi_0  
> nvidia-cublas-cu12        12.5.3.2                 pypi_0  
> nvidia-cuda-cupti-cu11    11.8.87                  pypi_0  
> nvidia-cuda-cupti-cu12    12.5.82                  pypi_0  
> nvidia-cuda-nvcc-cu12     12.5.82                  pypi_0  
> nvidia-cuda-nvrtc-cu11    11.8.89                  pypi_0  
> nvidia-cuda-nvrtc-cu12    12.5.82                  pypi_0  
> nvidia-cuda-runtime-cu11  11.8.89                  pypi_0  
> nvidia-cuda-runtime-cu12  12.5.82                  pypi_0  
> nvidia-cudnn-cu11         9.1.0.70                 pypi_0  
> nvidia-cudnn-cu12         9.3.0.75                 pypi_0  
> nvidia-cufft-cu11         10.9.0.58                pypi_0  
> nvidia-cufft-cu12         11.2.3.61                pypi_0  
> nvidia-curand-cu11        10.3.0.86                pypi_0  
> nvidia-curand-cu12        10.3.6.82                pypi_0  
> nvidia-cusolver-cu11      11.4.1.48                pypi_0  
> nvidia-cusolver-cu12      11.6.3.83                pypi_0  
> nvidia-cusparse-cu11      11.7.5.86                pypi_0  
> nvidia-cusparse-cu12      12.5.1.3                 pypi_0  
> nvidia-nccl-cu11          2.20.5                   pypi_0  
> nvidia-nccl-cu12          2.23.4                   pypi_0  
> nvidia-nvjitlink-cu12     12.5.82                  pypi_0  
> nvidia-nvtx-cu11          11.8.86                  pypi_0  
> oauthlib                  3.2.2                    pypi_0  
> opencv-python-headless    4.11.0.86                pypi_0  
> openssl                   3.0.16               h5eee18b_0  
> opt-einsum                3.4.0                    pypi_0  
> optax                     0.1.7                    pypi_0  
> optree                    0.14.1                   pypi_0  
> orbax-checkpoint          0.5.10                   pypi_0  
> packaging                 24.2                     pypi_0  
> pandas                    2.2.3                    pypi_0  
> parso                     0.8.4                    pypi_0  
> pexpect                   4.9.0                    pypi_0  
> pillow                    11.1.0                   pypi_0  
> pip                       25.0            py310h06a4308_0  
> platformdirs              4.3.7                    pypi_0  
> prompt-toolkit            3.0.50                   pypi_0  
> propcache                 0.3.1                    pypi_0  
> proto-plus                1.26.1                   pypi_0  
> protobuf                  5.29.4                   pypi_0  
> psutil                    7.0.0                    pypi_0  
> ptyprocess                0.7.0                    pypi_0  
> pure-eval                 0.2.3                    pypi_0  
> pyarrow                   19.0.1                   pypi_0  
> pyasn1                    0.6.1                    pypi_0  
> pyasn1-modules            0.4.2                    pypi_0  
> pydantic                  2.11.1                   pypi_0  
> pydantic-core             2.33.0                   pypi_0  
> pygments                  2.19.1                   pypi_0  
> python                    3.10.16              he870216_1  
> python-dateutil           2.9.0.post0              pypi_0  
> pytz                      2025.2                   pypi_0  
> pyyaml                    6.0.2                    pypi_0  
> readline                  8.2                  h5eee18b_0  
> regex                     2024.11.6                pypi_0  
> requests                  2.32.3                   pypi_0  
> requests-oauthlib         2.0.0                    pypi_0  
> rich                      14.0.0                   pypi_0  
> rsa                       4.9                      pypi_0  
> safetensors               0.5.3                    pypi_0  
> scipy                     1.12.0                   pypi_0  
> sentencepiece             0.2.0                    pypi_0  
> sentry-sdk                2.25.1                   pypi_0  
> setproctitle              1.3.5                    pypi_0  
> setuptools                75.8.0          py310h06a4308_0  
> simplejson                3.20.1                   pypi_0  
> simsimd                   6.2.1                    pypi_0  
> six                       1.17.0                   pypi_0  
> smmap                     5.0.2                    pypi_0  
> sniffio                   1.3.1                    pypi_0  
> sqlite                    3.45.3               h5eee18b_0  
> stack-data                0.6.3                    pypi_0  
> starlette                 0.46.1                   pypi_0  
> stringzilla               3.12.3                   pypi_0  
> sympy                     1.13.1                   pypi_0  
> tensorboard               2.19.0                   pypi_0  
> tensorboard-data-server   0.7.2                    pypi_0  
> tensorflow                2.19.0                   pypi_0  
> tensorflow-io-gcs-filesystem 0.37.1                   pypi_0  
> tensorstore               0.1.73                   pypi_0  
> termcolor                 3.0.1                    pypi_0  
> tiktoken                  0.9.0                    pypi_0  
> tk                        8.6.14               h39e8969_0  
> tokenizers                0.13.3                   pypi_0  
> tomli                     2.2.1                    pypi_0  
> toolz                     1.0.0                    pypi_0  
> torch                     2.4.0+cu118              pypi_0  
> torchaudio                2.4.0+cu118              pypi_0  
> torchvision               0.19.0+cu118             pypi_0  
> tqdm                      4.67.1                   pypi_0  
> traitlets                 5.14.3                   pypi_0  
> transformers              4.29.2                   pypi_0  
> treescope                 0.1.9                    pypi_0  
> triton                    3.0.0                    pypi_0  
> tux                       0.0.3                    pypi_0  
> typing-extensions         4.13.0                   pypi_0  
> typing-inspection         0.4.0                    pypi_0  
> tzdata                    2025.2                   pypi_0  
> urllib3                   2.3.0                    pypi_0  
> uvicorn                   0.34.0                   pypi_0  
> wandb                     0.19.9                   pypi_0  
> wcwidth                   0.2.13                   pypi_0  
> werkzeug                  3.1.3                    pypi_0  
> wheel                     0.45.1          py310h06a4308_0  
> wrapt                     1.17.2                   pypi_0  
> xxhash                    3.5.0                    pypi_0  
> xz                        5.6.4                h5eee18b_1  
> yarl                      1.18.3                   pypi_0  
> zipp                      3.21.0                   pypi_0  
> zlib                      1.2.13               h5eee18b_1 

There are two versions of cudnn,
> nvidia-cudnn-cu11         9.1.0.70                 pypi_0  
> nvidia-cudnn-cu12         9.3.0.75                 pypi_0 

And  I'm not quite sure which version of cudnn is correct


### 2. execute
Download the pre-trained model (see original readme)
executive command
```python -m latent_pretraining.inference```
The full output is as follows:
```
(lapaTemp) lyd@zkyd:/data/lyd/workspace/LatentVLA/LAPA$ python -m latent_pretraining.inference
CUDA backend failed to initialize: Unable to load cuDNN. Is it installed? (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
2025-04-03 15:53:10.229963: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1743666790.246398 1220611 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1743666790.251345 1220611 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1743666790.266092 1220611 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1743666790.266112 1220611 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1743666790.266119 1220611 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1743666790.266123 1220611 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
latent action is [[6 4 1 4]]
(lapaTemp) lyd@zkyd:/data/lyd/workspace/LatentVLA/LAPA$ 
```
!!!!!Finished!!!!!

Additionally, you may find situations where jax cannot call the GPU. For instance, I have to install cudnn91 to be able to call it. However, if cudnn91 is installed, it will conflict with the tux version (tux requires jax=0.4.23 & cudnn=89). In such cases, it can be installed separately[[ref]](https://blog.csdn.net/qq_52476897/article/details/134556167)
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda install cudnn==8.9.2.26
conda install cudatoolkit==11.8.0
```

# LAPA: Latent Action Pretraining from Videos
[[Project]](https://latentactionpretraining.github.io/)
[[Paper]](https://arxiv.org/abs/2410.11758)
[[Models]](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx)

**News**

[2025.01.22] LAPA has been accepted to ICLR 2025! 

[2024.11.22] We release the weights for the LAQ pretrained on openx here: https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/blob/main/laq_openx.pt. 

[2024.11.10] LAPA has won the [CoRL 2024 LangRob Workshop](https://sites.google.com/view/langrob-corl24/) **Best Paper Award** (among 75 accepted papers)! 🥳

**LAPA** 

- **Unsupervised approach** for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels.

- Outperforms the current state-of-the-art VLA model trained with ground-truth actions, building a new **SOTA VLA model**.

- Achieves over **30x** greater pretraining efficiency compared to conventional VLA pretraining.

<div align="center">
  <img src="./imgs/latent_action_pretraining.png"/>
</div>


## Getting Started 

```bash
conda create -n lapa python=3.10 -y
conda activate lapa
git clone https://github.com/LatentActionPretraining/LAPA.git
pip install -r requirements.txt 
mkdir lapa_checkpoints && cd lapa_checkpoints
```
Next, download the model checkpoint from [Huggingface](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx) repository. Download, three files under `lapa_checkpoints` directory. 

```bash
wget https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/tokenizer.model
wget https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/vqgan
wget https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/params
```

To run LAPA checkpoint which is pretrained on [Open-X Embodiment dataset](https://arxiv.org/abs/2310.08864), run the following command:
```bash
cd ..
python -m latent_pretraining.inference
```
This will generate the latent action conditioned on the input image and the natural language instruction.
You can change the input image and the instruction to a custom instance. **Note that the output space is the latent action space (which a space size of $8^4$), which is not the real action space**. To evaluate LAPA, fine-tuning is needed to map the latent space to the real action space (e.g. end-effector).

## Fine-tuning LAPA 
For fine-tuning LAPA on real world trajectories, you have to first preprocess the dataset to discretize the action space. We assume that there is a json file (`--input_path`) where the json file has the following row format:
```json
  {
    "id": "data/finetune_data/episode_0/step_0",
    "image": "data/finetune_data/episode_0/step_0.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat action should the robot take to `pick up the milk and put it in the sink`"
      },
      {
        "from": "gpt",
        "raw_actions": [
          0.0004934221506118809,
          -0.00011252239346504211,
          -0.001941084861755371,
          0.013634951062806884,
          0.013678191591275368,
          -0.004913635449167675,
          0.0
        ],
        "states": {
          "eef_pos": [
            0.24725835025310516,
            -0.022094586864113808,
            0.9283081889152527
          ],
          "eef_euler": [
            3.1202197128587876,
            -0.7113159765223936,
            -0.10937155062330725
          ],
          "gripper_state": 0.0
        }
      }
    ]
  }
```
where `finetune_data` contains the images of fine-tuning trajectories.

Run the following commands to preprocess the fine-tuning dataset and fine-tune LAPA.
```bash
python data/finetune_preprocess.py --input_path "/path_to_json_file" --output_filename "data/real_finetune.jsonl" --csv_filename "data/real_finetune.csv"
./scripts/finetune_real.sh
```
We ran the experiments with 4 80GB-A100 GPUs. To change the number of GPUs being used, change the second index of `--mesh_dim` in the script to the number of GPUs.

For fine-tuning on SIMPLER rollout trajectories (100 trajecories), run the following command:
```bash
./scripts/finetune_simpler.sh
```

After finetuning, to deploy the model, run the following command:
```bash
python -m latent_pretraining.deploy --load_checkpoint "params::/path_to_the_finetuned_ckpt" --action_scale_file "data/real_finetune.csv"
```
where `load_checkpoint` includes the path to the finet-uned checkpoint and `action_scale_file` includes the path to the csv file constructed during data preprocessing of fine-tuning dataset.
 
## Latent Action Quantization 
We provide the code for latent action quantization pretraining.
```bash
conda create -n laq python=3.10 -y
conda activate laq
cd laq
pip install -e .
accelerate launch train_sthv2.py
```
Note that the current data loader code is based on something-something v2 dataset structure where the directory consists of multiple trajectories and each trajectory contain multiple images. To train on custom dataset, either change the data structure or modify the existing data loading code. 

After training, you can use the trained quantization model as an inverse dynamics model to obtain latent actions for training data. 

```bash
python inference_sthv2.py
```
Add arguments based on the training arguements. For the `input_file` argument, it should be a jsonl file which contains `id`, `image`, `instruction` keys as the metadata and `vision` which is the output of the vqgan model consisting of 256 discrete image tokens as the otuput.


## Latent-Pretraining 
We provide the code to do latent pretraining from pretrained LWM checkpoint. First, download the [LWM-Chat-1M-Jax](https://huggingface.co/LargeWorldModel/LWM-Chat-1M-Jax) model under `lwm_checkpoints` directory. Then, download the pretraining dataset from this [link](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/latent_action_pretraining_openx.jsonl) under the `data` directory. Run the following command for latent pretraining:
```bash
./scripts/latent_pretrain_openx.sh
```
We experimented with 8 H100 GPUs for 34 hours. We have empirically observed that 70K steps with a batch size of 256 is enough to get decent performance on downstream tasks after fine-tuning.

## SIMPLER
As a reproducible simulation, we release the setup that we tested with. First, install packages required for our latent-pretraining and [SIMPLER](https://github.com/simpler-env/SimplerEnv) following the installation guide. 

The inference script is provided in `scripts/lapa_bridge.sh`.
## Acknowledgement 
The codebase is based on [Large-World-Model](https://github.com/LargeWorldModel/LWM) repository. For latent action quantization, we referred to [Phenaki](https://github.com/lucidrains/phenaki-pytorch) code. For deployment code, we referred to the [OpenVLA](https://github.com/openvla/openvla) code. For the SIMPLER evaluation code, we referred to the [SIMPLER](https://github.com/simpler-env/SimplerEnv) repository.


## Citation

If you use this codebase, or otherwise found our work valuable, please cite:
```
@article{ye2024latent,
  title={Latent Action Pretraining from Videos},
  author={Ye, Seonghyeon and Jang, Joel and Jeon, Byeongguk and Joo, Sejune and Yang, Jianwei and Peng, Baolin and Mandlekar, Ajay and Tan, Reuben and Chao, Yu-Wei and Lin, Bill Yuchen and others},
  journal={arXiv preprint arXiv:2410.11758},
  year={2024}
}
```

If you have additional question, feel free to send an email to latentactionpretraining@gmail.com.

## License

LAPA's code and model weights are released under the MIT License. 
