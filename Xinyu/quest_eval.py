
from pathlib import Path
import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from QuestEval.questeval.questeval_metric import QuestEval

MODEL_NAME = 't5-small'

questeval = QuestEval(task = 'summarization', do_weighter = True, no_cuda=False)
model = questeval.set_model('hyp','QG',MODEL_NAME)


source_1 = "Since 2000, the recipient of the Kate Greenaway medal has also been presented with the Colin Mears award to the value of 35000."
prediction_1 = "Since 2000, the winner of the Kate Greenaway medal has also been given to the Colin Mears award of the Kate Greenaway medal."
references_1 = [
    "Since 2000, the recipient of the Kate Greenaway Medal will also receive the Colin Mears Awad which worth 5000 pounds",
    "Since 2000, the recipient of the Kate Greenaway Medal has also been given the Colin Mears Award."
]

source_2 = "He is also a member of another Jungiery boyband 183 Club."
prediction_2 = "He also has another Jungiery Boyband 183 club."
references_2 = [
    "He's also a member of another Jungiery boyband, 183 Club.", 
    "He belonged to the Jungiery boyband 183 Club."
]

score = questeval.corpus_questeval(
    hypothesis=[model(source_1), prediction_2], 
    sources=[source_1, source_2],
    list_references=[references_1, references_2]
)

print(score)


