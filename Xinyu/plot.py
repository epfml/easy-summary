import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.3,0.9,4)
y = np.array([37.4,	38.59,	39.12,	39.71])

plt.plot(x,y,'o-')
plt.xlabel('summarization ratio of BERT')
plt.ylabel('SARI score')
plt.title('SARI score with summarization ratio of BERT')
plt.savefig('SARI_with_BERT_summarization_ratio.png')