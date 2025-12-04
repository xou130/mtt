from transformers import BertPreTrainedModel, BertModel
import torch

class CustomBertForSequenceClassification(BertPreTrainedModel):
    """
    Custom BERT classification model that supports direct input of embedding vectors
    
    This model allows direct passing of inputs_embeds instead of input_ids, suitable for training with pre-generated
    word embeddings or during knowledge distillation.
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)  # Load pre-trained BERT model
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)  # Classification layer
        self.init_weights()

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None):
        """
        Custom forward method that accepts inputs_embeds instead of input_ids

        Args:
            inputs_embeds: Directly passed token embeddings, shape [batch_size, seq_length, hidden_size]
            attention_mask: BERT attention mask, shape [batch_size, seq_length]
            labels: Labels for loss calculation, shape [batch_size]

        Returns:
            Tuple (loss, logits) or just logits (if labels not provided)
        """
        
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")
        
        outputs = self.bert(
            inputs_embeds=inputs_embeds,  # Use embeddings as input
            attention_mask=attention_mask,
        )
        pooled_output = outputs[1]  # [CLS] token representation
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)
