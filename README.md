# Nash Model-Agnostic Meta-Learning
## Penalty type
### Default
$\mathcal{L}_{i}$
### Penalty type 1
$p_{1}(w)={{w}\over{dim(\phi)}} \left({{N}\over{\alpha B}}\right)^{2}$ 

### Penalty type 2
$p_{2}(x)$

## Sinusoid Regression
### 5-shot sinusoid MAML train
python main.py --datasource=1 --model_type=1 --penalty_type=0 --train --seed=5
### 5-shot sinusoid MAML test
python main.py --datasource=1 --model_type=1 --penalty_type=0

### 5-shot sinusoid MAML+p1(1) train
python main.py --datasource=1 --model_type=1 --penalty_type=1 --train --seed=5 --weight1=1.0
### 5-shot sinusoid MAML+p1(1) test
python main.py --datasource=1 --model_type=1 --penalty_type=1

### 5-shot sinusoid MAML+p2(0.5, 0.0000001) train
python main.py --datasource=1 --model_type=1 --penalty_type=2 --train --seed=5 --weight2=0.5 --weight3=0.0000001
### 5-shot sinusoid MAML+p2(0.5, 0.0000001) test
python main.py --datasource=1 --model_type=1 --penalty_type=2


### 5-shot sinusoid NashCAVIA3 train
python main.py --datasource=1 --model_type=5 --penalty_type=3 --train --seed=6 --num_context_params=4

### 5-shot sinusoid NashCAVIA3 test
python main.py --datasource=1 --model_type=5 --penalty_type=3 --num_context_params=4


