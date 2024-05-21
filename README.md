# Nash Gradient-Based Meta-Learning
## Penalty term
### Penalty function
$$p_{1}(w)=w \left({{B}\over{\alpha N}}\right)^{2} \lVert \theta - {{1}\over{B}}\sum\limits_{k\ne i}{\phi_{k}} - {{1}\over{B}} \phi_{i} \rVert_{2}^{2}$$

where $\alpha$ is inner learning rate, $N$ is the number of task, $B$ is the batch-size, and $\dim(\phi)$ is the number of parameters.

$$p_{2}(w, r) = w {{B \lVert \theta - \phi_{i} \rVert_{2}^{r}}\over{C + \sum\limits_{k=1}^{B}{\lVert \theta - \phi_{k} \rVert_{2}^{r}}}}$$

where $N$ is the number of task, $B$ is the batch-size, and $C$ is a very small constant to prevent division by zero.

### Penalty 1 with weight $w$: $p_{1}(w)$
### Penalty 2 with weight $w$: $p_{2}(w, 2)$
### Penalty 3 with weight $w$: $p_{2}(w, 4)$


## Sinusoid Regression - MAML
### 5-shot sinusoid MAML train:
python main.py --datasource=1 --model_type=1 --penalty_type=0 --train
### 5-shot sinusoid MAML test:
python main.py --datasource=1 --model_type=1 --penalty_type=0

### 5-shot sinusoid MAML + Penalty 1 with weight $1.0$ train:
python main.py --datasource=1 --model_type=1 --penalty_type=1 --train --weight1=1.0
### 5-shot sinusoid MAML + Penalty 1 with weight $1.0$ test:
python main.py --datasource=1 --model_type=1 --penalty_type=1

### 5-shot sinusoid MAML + Penalty 2 with weight $0.5$ train:
python main.py --datasource=1 --model_type=1 --penalty_type=2 --train --weight2=0.5 --weight3=0.0000001
### 5-shot sinusoid MAML + Penalty 2 with weight $0.5$ test:
python main.py --datasource=1 --model_type=1 --penalty_type=2

### 5-shot sinusoid MAML + Penalty 3 with weight $0.5$ train:
python main.py --datasource=1 --model_type=1 --penalty_type=3 --train --weight2=0.5 --weight3=0.00000000000001
### 5-shot sinusoid MAML + Penalty 3 with weight $0.5$ test:
python main.py --datasource=1 --model_type=1 --penalty_type=3


## Sinusoid Regression - TR-MAML
### 5-shot sinusoid TR-MAML train:
python main.py --datasource=1 --model_type=2 --penalty_type=0 --p_lr=0.00001 --train
### 5-shot sinusoid TR-MAML test:
python main.py --datasource=1 --model_type=2 --penalty_type=0

### 5-shot sinusoid TR-MAML + Penalty 1 with weight $1.0$ train:
python main.py --datasource=1 --model_type=2 --penalty_type=1 --p_lr=0.00001 --train --weight1=1.0
### 5-shot sinusoid TR-MAML + Penalty 1 with weight $1.0$ test:
python main.py --datasource=1 --model_type=2 --penalty_type=1

### 5-shot sinusoid TR-MAML + Penalty 2 with weight $0.5$ train:
python main.py --datasource=1 --model_type=2 --penalty_type=2 --p_lr=0.00001 --train --weight2=0.5 --weight3=0.0000001
### 5-shot sinusoid TR-MAML + Penalty 2 with weight $0.5$ test:
python main.py --datasource=1 --model_type=2 --penalty_type=2

### 5-shot sinusoid TR-MAML + Penalty 3 with weight $0.5$ train:
python main.py --datasource=1 --model_type=2 --penalty_type=3 --p_lr=0.00001 --train --weight2=0.5 --weight3=0.00000000000001
### 5-shot sinusoid TR-MAML + Penalty 3 with weight $0.5$ test:
python main.py --datasource=1 --model_type=2 --penalty_type=3


## Sinusoid Regression - Meta-SGD
### 5-shot sinusoid Meta-SGD train:
python main.py --datasource=1 --model_type=4 --penalty_type=0 --train
### 5-shot sinusoid Meta-SGD test:
python main.py --datasource=1 --model_type=4 --penalty_type=0

### 5-shot sinusoid Meta-SGD + Penalty 1 with weight $1.0$ train:
python main.py --datasource=1 --model_type=4 --penalty_type=1 --train --weight1=1.0
### 5-shot sinusoid Meta-SGD + Penalty 1 with weight $1.0$ test:
python main.py --datasource=1 --model_type=4 --penalty_type=1

### 5-shot sinusoid Meta-SGD + Penalty 2 with weight $0.5$ train:
python main.py --datasource=1 --model_type=4 --penalty_type=2 --train --weight2=0.5 --weight3=0.0000001
### 5-shot sinusoid Meta-SGD + Penalty 2 with weight $0.5$ test:
python main.py --datasource=1 --model_type=4 --penalty_type=2

### 5-shot sinusoid Meta-SGD + Penalty 3 with weight $0.5$ train:
python main.py --datasource=1 --model_type=4 --penalty_type=3 --train --weight2=0.5 --weight3=0.00000000000001
### 5-shot sinusoid Meta-SGD + Penalty 3 with weight $0.5$ test:
python main.py --datasource=1 --model_type=4 --penalty_type=3


## Sinusoid Regression - CAVIA
### 5-shot sinusoid CAVIA train:
python main.py --datasource=1 --model_type=5 --penalty_type=0 --train --num_context_params=4
### 5-shot sinusoid CAVIA test:
python main.py --datasource=1 --model_type=5 --penalty_type=0 --num_context_params=4

### 5-shot sinusoid CAVIA + Penalty 1 with weight $1.0$ train:
python main.py --datasource=1 --model_type=5 --penalty_type=1 --train --weight1=1.0 --num_context_params=4
### 5-shot sinusoid CAVIA + Penalty 1 with weight $1.0$ test:
python main.py --datasource=1 --model_type=5 --penalty_type=1 --num_context_params=4

### 5-shot sinusoid CAVIA + Penalty 2 with weight $0.5$ train:
python main.py --datasource=1 --model_type=5 --penalty_type=2 --train --weight2=0.5 --weight3=0.0000001 --num_context_params=4
### 5-shot sinusoid CAVIA + Penalty 2 with weight $0.5$ test:
python main.py --datasource=1 --model_type=5 --penalty_type=2 --num_context_params=4

### 5-shot sinusoid CAVIA + Penalty 3 with weight $0.5$ train:
python main.py --datasource=1 --model_type=5 --penalty_type=3 --train --weight2=0.5 --weight3=0.00000000000001 --num_context_params=4
### 5-shot sinusoid CAVIA + Penalty 3 with weight $0.5$ test:
python main.py --datasource=1 --model_type=5 --penalty_type=3 --num_context_params=4

## Classification
If you need the Mini-ImageNet dataset, please follow https://github.com/yaoyao-liu/mini-imagenet-tools

Put them in the folder "data/miniimagenet". The label files are already in there.

To run the experiment, set datasource value as 2. For the other argument options, see arguments.py

python main.py --datasource=2
