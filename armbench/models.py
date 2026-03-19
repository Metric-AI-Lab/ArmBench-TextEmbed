"""Embedding model wrapper for SentenceTransformer and HuggingFace models."""

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling, Normalize
from huggingface_hub import HfFileSystem


class Embedding_Model:
    """Unified embedding model wrapper supporting SentenceTransformer and HuggingFace models."""

    def __init__(self, model_name, pooling=None, max_length=512,batch_size=32, **kwargs):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.is_sentence_transformer = self._is_sentence_transformer(model_name)

        if self.is_sentence_transformer:           
            self.model = SentenceTransformer(model_name,model_kwargs={"trust_remote_code": True}, **kwargs)
        else:
            valid_pooling = ('mean', 'last_token', 'cls_token')
            if pooling is None:
                raise ValueError(f"'pooling' must be specified for non-SentenceTransformer models. Use one of: {valid_pooling}")
            if pooling not in valid_pooling:
                raise ValueError(f"Invalid pooling type '{pooling}'. Must be one of: {valid_pooling}")

            self.model = self._wrap_hf_model(model_name, pooling, **kwargs)

        self.model.max_seq_length = max_length

    def _is_sentence_transformer(self, model_name):
        """Check if a model is a SentenceTransformer by looking for modules.json."""
        try:
            fs = HfFileSystem()
            return fs.exists(f"{model_name}/modules.json")
        except Exception:
            return False

    def _wrap_hf_model(self, model_name, pooling, **kwargs):
        """Wrap a HuggingFace model with pooling and normalization layers."""
        pooling_mode = {
            'mean': 'mean_tokens',
            'cls_token': 'cls_token',
            'last_token': 'lasttoken'
        }.get(pooling, 'mean_tokens')

        transformer = Transformer(model_name, **kwargs)
        pooling_layer = Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_cls_token=(pooling_mode == 'cls_token'),
            pooling_mode_mean_tokens=(pooling_mode == 'mean_tokens'),
            pooling_mode_lasttoken=(pooling_mode == 'lasttoken')
        )
        normalize_layer = Normalize()
        return SentenceTransformer(modules=[transformer, pooling_layer, normalize_layer])

    def encode(self, sentences, show_progress=True, **kwargs):
        """Encode sentences into embeddings."""
        return self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            **kwargs
        )
