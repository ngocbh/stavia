from stavia import log_linear_model as llm
from stavia import logistic_regression as lr
from stavia.crf import trainer
from stavia.utils.parameters import *
#test rsync
# trainer.train_crf()
print('METHOD=',METHOD)
print('MODEL_ID=',MODEL_ID)
if METHOD == 'llm':
	llm.train()
else:
	lr.train()
