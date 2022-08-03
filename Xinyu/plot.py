import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.1,0.9,5)
y = np.array([36.48, 38.37,	38.79,	39.55,	39.74])

plt.plot(x,y,'o-')
plt.xlabel('summarization ratio of BERT')
plt.ylabel('SARI score')
plt.title('SARI score with summarization ratio of BERT')
plt.savefig('SARI_with_BERT_summarization_ratio.png')