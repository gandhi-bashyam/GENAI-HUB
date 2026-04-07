from common.ranking.normalization import z_score_normalize, softmax
from common.ranking.fusion import reciprocal_rank_fusion

print("Z-score:", z_score_normalize([10, 20, 30, 40]))
print("Softmax:", softmax([1, 2, 3]))

bm25 = ["doc1", "doc2", "doc3"]
vector = ["doc2", "doc3", "doc4"]

print("Fusion:", reciprocal_rank_fusion([bm25, vector]))