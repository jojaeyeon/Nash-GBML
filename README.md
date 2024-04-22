# Nash Model-Agnostic Meta-Learning
## Penalty term
### Penalty function
$$p_{1}(w)={{w}\over{\dim(\phi)}} \left({{N}\over{\alpha B}}\right)^{2} \lVert \theta - {{1}\over{N}}\sum\limits_{k\ne i}{\phi_{k}} - {{1}\over{N}} \phi_{i} \rVert_{2}^{2}$$

where $\alpha$ is inner learning rate, $N$ is the number of task, $B$ is the batch-size, and $\dim(\phi)$ is the number of parameters.

$$p_{2}(w, r) = w {{B \lVert \theta - \phi_{i} \rVert_{2}^{r}}\over{C + \sum\limits_{k=1}^{N}{\lVert \theta - \phi_{k} \rVert_{2}^{r}}}}$$

where $N$ is the number of task, $B$ is the batch-size, and $C$ is a very small constant to prevent division by zero.

### Penalty 1 (w): $p_{1}(w)$
### Penalty 2 (w): $p_{2}(w, 2)$
### Penalty 3 (w): $p_{2}(w, 4)$

## Sinusoid Regression
### 5-shot sinusoid MAML train:
python main.py --datasource=1 --model_type=1 --penalty_type=0 --train --seed=5
### 5-shot sinusoid MAML test:
python main.py --datasource=1 --model_type=1 --penalty_type=0

### 5-shot sinusoid MAML+$p_{1}\left(1.0\right)$ train:
python main.py --datasource=1 --model_type=1 --penalty_type=1 --train --seed=5 --weight1=1.0
### 5-shot sinusoid MAML+$p_{1}\left(1.0\right)$ test:
python main.py --datasource=1 --model_type=1 --penalty_type=1

### 5-shot sinusoid MAML+$p_{2}\left(0.5, 2\right)$ train:
python main.py --datasource=1 --model_type=1 --penalty_type=2 --train --seed=5 --weight2=0.5 --weight3=0.0000001
### 5-shot sinusoid MAML+$p_{2}\left(0.5, 2\right)$ test:
python main.py --datasource=1 --model_type=1 --penalty_type=2

### 5-shot sinusoid MAML+$p_{2}\left(0.5, 4\right)$ train:
python main.py --datasource=1 --model_type=1 --penalty_type=3 --train --seed=5 --weight2=0.5 --weight3=0.00000000000001
### 5-shot sinusoid MAML+$p_{2}\left(0.5, 4\right)$ test:
python main.py --datasource=1 --model_type=1 --penalty_type=3




### 5-shot sinusoid NashCAVIA3 train:
python main.py --datasource=1 --model_type=5 --penalty_type=3 --train --seed=6 --num_context_params=4

### 5-shot sinusoid NashCAVIA3 test:
python main.py --datasource=1 --model_type=5 --penalty_type=3 --num_context_params=4


